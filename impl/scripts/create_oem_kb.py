import argparse
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import os
import concurrent.futures
import time
import json
import glob
import threading
import shutil

from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.config import Settings

from transformers import AutoTokenizer
import requests

# from components.encryption.crypto_handler import CryptoHandler
# from config import AMDCHAT_DATA_KEY_GUID

import hashlib

EMBEDDING_SERVERS = ["http://127.0.0.1:1234/v1"]
MODEL = "local"
DOC_CHUNK_SIZE = 500
DOC_CHUNK_OVERLAP = 50
QNA_CHUNK_SIZE = 200
QNA_CHUNK_OVERLAP = 0
DB_FILE_NAME = "Chroma_oem"
DB_COLLECTION_NAME = "vector_store_chroma_oem"
BATCH_SIZE = 32
LANGUAGE = "en-US"

# version is formated as date string YYYYMMDD.H
KB_VERSION = time.strftime("%Y%m%d.%H")

CHAT_LOCALAPPDATA_PATH = os.path.expandvars("%LOCALAPPDATA%/AMD/CN/Chat")
VECTOR_STORE_PATH = os.path.join(CHAT_LOCALAPPDATA_PATH, "vector_store")

def join_and_make_path(*args):
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def init_config(lang, server_url, api_key):
    global EMBEDDING_SERVERS, EMBEDDING_API_KEY, DOC_CHUNK_SIZE, DOC_CHUNK_OVERLAP, QNA_CHUNK_SIZE, QNA_CHUNK_OVERLAP, BATCH_SIZE, LANGUAGE

    with open("./impl/configs/kb_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        LANGUAGE = lang
        EMBEDDING_SERVERS = config[lang]["servers"] if server_url is None else [server_url]
        EMBEDDING_API_KEY = config[lang]["api_key"] if api_key is None else api_key
        DOC_CHUNK_SIZE = config[lang]["doc_chunk_size"]
        DOC_CHUNK_OVERLAP = config[lang]["doc_chunk_overlap"]
        QNA_CHUNK_SIZE = config[lang]["qna_chunk_size"]
        QNA_CHUNK_OVERLAP = config[lang]["qna_chunk_overlap"]
        BATCH_SIZE = config[lang]["batch_size"]

    print(
        f"Config: {LANGUAGE} {EMBEDDING_SERVERS} {DOC_CHUNK_SIZE} {DOC_CHUNK_OVERLAP} {QNA_CHUNK_SIZE} {QNA_CHUNK_OVERLAP} {BATCH_SIZE}"
    )


class AvailableServer:
    def __init__(self):
        self.url = None

    def __enter__(self):
        while True:
            try:
                self.url = EMBEDDING_SERVERS.pop(0)
                if LANGUAGE == "zh-CN":
                    EMBEDDING_SERVERS.append(self.url)
                return self.url
            except IndexError:
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        EMBEDDING_SERVERS.append(self.url)


bad_embeddings = 0
tokenizer_global = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
tokenizer_china = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")


def token_len(text: str) -> int:
    tokenizer = tokenizer_china if LANGUAGE == "zh-CN" else tokenizer_global
    encoded_input = tokenizer(text, padding=True, truncation=False)
    autotokenizer_len = len(encoded_input.data["input_ids"])
    return autotokenizer_len


def token_len_server(text: str):
    with AvailableServer() as server:
        embeddings = requests.post(f"{server}tokenize", json={"content": text}).json()
        return len(embeddings["tokens"])


class EmbeddingServerQueue(Embeddings):
    def queue_batch(self, texts):
        global bad_embeddings
        while True:
            try:
                with AvailableServer() as server:
                    client = OpenAI(base_url=f"{server}", api_key=EMBEDDING_API_KEY)
                    embeddings = client.embeddings.create(
                        model=MODEL,
                        input=texts,
                    )

                    results = []
                    dimension = 1536 if LANGUAGE == "zh-CN" else 1024
                    # for emb in embeddings.data:
                    #     len_embedding = len(emb.embedding)
                    #     if (
                    #         len_embedding > 0 and len_embedding <= 1536
                    #         if LANGUAGE == "zh-CN"
                    #         else 1024
                    #     ):
                    #         dimension = len_embedding
                    #         break
                    for emb in embeddings.data:
                        if len(emb.embedding) == 0 or emb.embedding[0] is None:
                            bad_embeddings += 1
                            results.append([0.0] * dimension)
                        else:
                            results.append(emb.embedding)

                    return results
            except Exception as e:
                # print(f"queue_batch {e}")
                pass

    def embed_documents(self, texts):
        batch_size = BATCH_SIZE
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(self.queue_batch, batch))

            with tqdm(total=len(futures), desc="Embedding") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)

            # make sure we read the results in order
            for future in futures:
                results.extend(future.result())

        return results

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def embed_documents(documents, output_path):
    db = Chroma.from_documents(
        documents=documents,
        embedding=EmbeddingServerQueue(),
        persist_directory=output_path,
        collection_name=DB_COLLECTION_NAME,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
        collection_metadata={"hnsw:space": "cosine"},
    )


def process_document(file, folder, refined_folder):
    with open(file, "r", encoding="utf8") as f:
        content = f.read()

    # 100k limit
    if len(content) > 100000:
        return []

    import pathlib

    doc_splitter = MarkdownTextSplitter(
        chunk_size=DOC_CHUNK_SIZE,
        chunk_overlap=DOC_CHUNK_OVERLAP,
        length_function=token_len,
        add_start_index=True,
    )

    if token_len(content) > DOC_CHUNK_SIZE:
        doc_pieces = doc_splitter.split_text(content)
    else:
        doc_pieces = [content]

    metadata = {
        "hnsw:space": "cosine",
        "source": os.path.join(folder, Path(file).name),
        "type": "document",
    }

    if refined_folder:
        tag_file = os.path.join(refined_folder, Path(file).stem + "_tags.json")
        if os.path.exists(tag_file):
            # print(f"found Tags file {tag_file}")
            try:
                with open(tag_file, "r", encoding="utf8") as tf:
                    tag_json = json.loads(tf.read())
                    url = tag_json.get("url", "")
                    title = tag_json.get("title", "")
                    tags = tag_json.get("tags", [])
                    updated_date = tag_json.get("updated_date", "")
                    publised_date = tag_json.get("published_date", "")
                    copyright_year = tag_json.get("copyright_year", "")
                    metadata["tags"] = str(tags)
                    metadata["url"] = url
                    metadata["title"] = title
                    metadata["updated_date"] = updated_date
                    metadata["published_date"] = publised_date
                    metadata["copyright_year"] = copyright_year
            except Exception as e:
                print(f"Error loading {tag_file}")
                print(e)

    documents = []
    for doc in doc_pieces:
        documents.append(Document(page_content=doc, metadata=metadata))

    if refined_folder:
        qna_file = os.path.join(refined_folder, Path(file).stem + "_qna.txt")
        if os.path.exists(qna_file):
            # print(f"found Q&A file {qna_file}")
            metadata_qna = metadata.copy()
            metadata_qna["type"] = "q&a"

            qna_splitter = MarkdownTextSplitter(
                chunk_size=QNA_CHUNK_SIZE,
                chunk_overlap=QNA_CHUNK_OVERLAP,
                length_function=token_len,
                add_start_index=True,
            )

            with open(qna_file, "r", encoding="utf8") as f:
                qna_content = f.read()

                if token_len(qna_content) > QNA_CHUNK_SIZE:
                    qna_pieces = qna_splitter.split_text(qna_content)
                else:
                    qna_pieces = [qna_content]

                for qna in qna_pieces:
                    documents.append(Document(page_content=qna, metadata=metadata_qna))

        summary_file = os.path.join(refined_folder, Path(file).stem + "_summary.txt")
        if os.path.exists(summary_file):
            # print(f"found Summary file {summary_file}")
            metadata_summary = metadata.copy()
            metadata_summary["type"] = "summary"

            with open(summary_file, "r", encoding="utf8") as f:
                summary_content = f.read()
                if token_len(summary_content) > DOC_CHUNK_SIZE:
                    summary_pieces = doc_splitter.split_text(summary_content)
                else:
                    summary_pieces = [summary_content]

                for summary in summary_pieces:
                    documents.append(
                        Document(page_content=summary, metadata=metadata_summary)
                    )

    return documents


def process_folder(folder: str, refined_path: str):
    print(f"Processing {folder}")

    if os.path.isfile(folder):
        return process_document(folder, "", refined_path)

    refined_folder = os.path.join(refined_path, Path(folder).name) if refined_path else None

    files = glob.glob(folder + "**/**", recursive=True)
    documents = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for file in files:
            if os.path.isfile(file):
                if os.path.basename(file) == "referer_url.txt":
                    continue

                if "amd-documentation" in folder:
                    if not file.endswith(".md"):
                        continue

                futures.append(
                    executor.submit(process_document, file, folder, refined_folder)
                )

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                if r:
                    documents.extend(r)
                pbar.update(1)

    return documents


def generate_kb(lang, documents_path, refined_path, output_path, server_url, api_key):
    init_config(lang, server_url, api_key)

    documents = []
    for folder in os.listdir(documents_path):
        d = process_folder(os.path.join(documents_path, folder), refined_path)
        if d:
            documents.extend(d)

    # output_db_path = join_and_make_path(
    #     output_path, "kb", "update", KB_VERSION, DB_FILE_NAME
    # )
    output_db_path = output_path

    print(f"Embedding {len(documents)} documents")
    embed_documents(documents, output_db_path)
    print(f"Bad embeddings: {bad_embeddings}")
