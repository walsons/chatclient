from tqdm import tqdm
from pathlib import Path
import os
from langchain_text_splitters import MarkdownTextSplitter
import concurrent.futures
import time
import json
import glob
from transformers import AutoTokenizer
from typing import List
from pydantic import BaseModel
import threading
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


SYSTEM_PROMPT = """
You're an AI assistant who likes to follow the rules and follow the user's instructions, 
generating high-quality responses that satisfy the user based on the given content
"""

QNA_PROMPT = """
Task:
Your task is to create a Question and Answer (Q&A) document based on the given content.

# Rules that must be followed:
1: Ensure that each answer accurately uses information from the given content.
2: Ensure each question addresses a different information point within the given content to avoid repetition.
3: Ensure all answers must be derived solely from the given content; do not include any content from URLs or external sources.
4: Ensure that each question generated has a clear subject or target of enquiry and can be asked individually to anyone without causing misunderstanding.
5: Ensure that each generated question should not contain ambiguous objects such as this title, this product, this resource, this content, this page, this document, this article.
6: Ensure that generated questions are not related to copyright information and support language.
7: The format of each Question and Answer pair is as follows:
'''
QUESTION: Put the question here.
ANSWER: Put your answer to the above question here.
'''

8: If the given content does not provide sufficient information to answer a question and cannot be answered, use "N/A".
9: If you are unable to create a Q&A document, simply return "Question: N/A".

# Follow above rules and execute the below steps one by one:
step 1: Carefully read the entire given content to understand its details.
Step 2: Develop the first question and answer based on the URL and title name in the given content, the first question should mainly ask for information about the URL of the title name.
step 3: Based on the rules defined in above, continue to develop additional pairs of questions and answers that cover all the information in the given content, the number should between 10 to 50.
step 4: Check and improve all questions and answers to improve quality, avoid ambiguity, and make them easy to understand.
step 5: Do not explain your answer, you only need to return all the generated Question and Answer (Q&A) pairs to me without generating any other redundant descriptive text!
"""

QNA_PROMPT_CN = """
Task:
Your task is to create a Question and Answer (Q&A) document based on the given content.

# Rules that must be followed:
1: Respond in Chinese as I cannot read English.
2: Ensure that each answer accurately uses information from the given content.
3: Ensure each question addresses a different information point within the given content to avoid repetition.
4: Ensure all answers must be derived solely from the given content; do not include any content from URLs or external sources.
5: Ensure that each question generated has a clear subject or target of enquiry and can be asked individually to anyone without causing misunderstanding.
6: Ensure that each generated question should not contain ambiguous objects such as this title, this product, this resource, this content, this page, this document, this article.
7: Ensure that generated questions are not related to copyright information and support language.
8: The format of each Question and Answer pair is as follows:
'''
QUESTION: Put the question here.
ANSWER: Put your answer to the above question here.
'''

9: If the given content does not provide sufficient information to answer a question and cannot be answered, use "N/A".
10: If you are unable to create a Q&A document, simply return "Question: N/A".

# Follow above rules and execute the below steps one by one:
step 1: Carefully read the entire given content to understand its details.
Step 2: Develop the first question and answer based on the URL and title name in the given content, the first question should mainly ask for information about the URL of the title name.
step 3: Based on the rules defined in above, continue to develop additional pairs of questions and answers that cover all the information in the given content, the number should between 10 to 50.
step 4: Check and improve all questions and answers to improve quality, avoid ambiguity, and make them easy to understand.
step 5: Do not explain your answer, you only need to return all the generated Question and Answer (Q&A) pairs to me without generating any other redundant descriptive text!
"""

SUMMARY_PROMPT = """
Your task is to read the given content and produce a concise summary based on it. 
The summary should be concise, capturing the key points within 50 to 200 words. 
Ensure that the summary is solely based on the document's content and does not include any information from external sources or URLs. 
If summarization is not possible, respond with "Summary: N/A".
Do not explain your answer, you only need to return the summary content to me without generating any other redundant descriptive text!
"""

SUMMARY_PROMPT_CN = """
Your task is to read the given content and produce a concise summary based on it. 
The summary should be concise, capturing the key points within 50 to 200 words. 
Ensure that the summary is solely based on the document's content and does not include any information from external sources or URLs. 
If summarization is not possible, respond with "Summary: N/A".
Respond in Chinese as I cannot read English.
Do not explain your answer, you only need to return the summary content to me without generating any other redundant descriptive text!
"""

TAG_PROMPT = """
Your task is to take a look at this document, and create a list of tags based on the main content of the document.

Some example of tags and their descriptions are:
software - the document is about amd software features
hardware - the document is about amd hardware products
cpu - the document is about amd CPU
gpu - the document is about amd GPU
consumer - the document is about consumer products
enterprise - the document is about enterprise products
server - the document is about server products
troubleshooting - the document is helping user troubleshoot an issue
installation - the document is about driver installation issues
performance - the document is about performance of amd products
compatibility - the document is about software or hardware compatibility with amd products
gaming - the document is about gaming features or performance
security - the document is about security features or issues
features - the document is about features of amd products
howto - the document is a how-to guide
adrenalin software - the document is about "adrenalin software" or "amd software adrenalin edition"
radeon settings - include this tag only if the document is about "Radeon Settings" or "Radeon Software Crimson", do not include for document about "adrenalin software"
recommendations - the document provides recommendations for amd products
upgrade - the document is about upgrading amd products
release notes - the document is about release notes for amd products or drivers

add additional tags for any PC games mentioned in the document, use game title as the tag
add additional tags for any specific feature or product that's focused in the document, use the feature or product name as the tag
if the document is about a specific version of software, driver or feature release, add a tag for the software, driver, feature name, and add a tag for the version number

choose the tags that best describe the content of the document and return in this format:
example
["adrenalin software", "features"]

If you cannot tag this document, simply return []
"""

"""
Global param
"""

LANGUAGE = "en-US"
OUTPUT_FILES = []
lock = threading.Lock()


class Config(BaseModel):
    servers: List[str]
    api_key: str
    model: str
    context: int
    chunck_size: int
    generation_size: int
    regenerate_tags: bool
    qa_temperature: float
    workers: int


CONFIG: Config
tokenizer = None


class AvailableServer:
    def __init__(self):
        self.url = None

    def __enter__(self):
        while True:
            try:
                self.url = CONFIG.servers.pop(0)
                return self.url
            except IndexError:
                time.sleep(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        CONFIG.servers.append(self.url)


def send_completions_local(messages, temperature=0.1) -> str:
    while True:
        try:
            with AvailableServer() as server:
                client = ChatOpenAI(
                    base_url=server,
                    api_key=CONFIG.api_key,
                    model=CONFIG.model,
                    temperature=temperature,
                    max_tokens=CONFIG.generation_size,
                    extra_body={"repeat_penalty": 1.1},
                )

                response = client.invoke(messages).content
                return response
        except Exception as e:
            print(e)
            pass


def send_completions(messages, temperature=0.1) -> str:
    with lock:
        return send_completions_local(messages, temperature)


def template_messages(task_prompt: str, content: str):
    with lock:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"{task_prompt}\nBelow is the given content you need to go over and review carefully: \n```\n{content}\n```\n",
            ),
        ]
        return messages


def token_len(text: str) -> int:
    if tokenizer is not None:
        encoded_input = tokenizer(text, truncation=False)
        return len(encoded_input.data["input_ids"])

    # Since we don't know the type of user's LLM model, we use token-count to approximately calculate the token length
    from token_count import TokenCount
    tc = TokenCount()
    return tc.num_tokens_from_string(text)


def create_qna(output_path, file, content_pieces):
    output_file = os.path.join(output_path, Path(file).stem + "_qna.txt")
    if os.path.exists(output_file):
        OUTPUT_FILES.append(output_file)
        return

    results = []

    qa_prompt = QNA_PROMPT if LANGUAGE == "en-US" else QNA_PROMPT_CN
    for content in content_pieces:
        messages = template_messages(qa_prompt, content)
        result = send_completions(messages, temperature=CONFIG.qa_temperature)
        results.append(result)

    # write to file
    with open(output_file, "w", encoding="utf8") as f:
        OUTPUT_FILES.append(output_file)
        f.write("\n".join(results))


def create_summary(output_path, file, content_pieces):
    output_file = os.path.join(output_path, Path(file).stem + "_summary.txt")
    if os.path.exists(output_file):
        OUTPUT_FILES.append(output_file)
        return

    results = []

    sm_prompt = SUMMARY_PROMPT if LANGUAGE == "en-US" else SUMMARY_PROMPT_CN
    for content in content_pieces:
        messages = template_messages(sm_prompt, content)
        result = send_completions(messages)
        results.append(result)

    # write to file
    with open(output_file, "w", encoding="utf8") as f:
        OUTPUT_FILES.append(output_file)
        f.write("\n".join(results))


def create_tags(output_path, file, content_pieces):
    output_file = os.path.join(output_path, Path(file).stem + "_tags.json")

    tags = set()

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf8") as f:
            file_json = json.load(f)
            if "tags" in file_json:
                file_tags = list(filter(None, file_json["tags"]))
            tags.update(file_tags)

    if CONFIG.regenerate_tags:
        llm_results = []

        for content in content_pieces:
            messages = template_messages(TAG_PROMPT, content)
            result = send_completions(messages)
            llm_results.append(result)

        llm_final = "\n".join(filter(None, llm_results))

        # remove duplicate tags
        for line in llm_final.split("\n"):
            clean = line.strip().replace("[", "").replace("]", "").replace('"', "")
            for tag in clean.split(","):
                tags.add(tag.strip().lower())

    tag_json = {
        "tags": sorted(filter(None, tags)),
    }
    meta = ["url", "title", "updated_date", "published_date", "copyright_year"]
    # add tag for url
    for content in content_pieces:
        for line in content.split("\n"):
            if line.startswith("Url:"):
                link = line.split("Url:")[1].strip()
                tag_json["url"] = link
            if line.startswith("Title:"):
                title = line.split("Title:")[1].strip()
                tag_json["title"] = title
            if line.startswith("Updated Date:"):
                date = line.split("Updated Date:")[1].strip()
                tag_json["updated_date"] = date
            if line.startswith("Published Date:"):
                date = line.split("Published Date:")[1].strip()
                tag_json["published_date"] = date
            if line.startswith("Copyright Year:"):
                date = line.split("Copyright Year:")[1].strip()
                tag_json["copyright_year"] = date

            if all(tag_json.get(m) for m in meta):
                break
    with open(output_file, "w", encoding="utf8") as f:
        OUTPUT_FILES.append(output_file)
        f.write(
            json.dumps(
                tag_json, indent=4, ensure_ascii=True if LANGUAGE == "en-US" else False
            )
        )


def refine_file(output_path, file):
    with open(file, "r", encoding="utf8") as f:
        content = f.read()

    # 100k limit
    if len(content) > 100000:
        return file

    text_splitter = MarkdownTextSplitter(
        chunk_size=CONFIG.chunck_size,
        chunk_overlap=0,
        length_function=token_len,
        add_start_index=True,
    )

    content_pieces = text_splitter.split_text(content)
    create_tags(output_path, file, content_pieces)
    create_qna(output_path, file, content_pieces)
    create_summary(output_path, file, content_pieces)
    return file


def refine(
    folder: str,
    output_folder: str = "refined",
):
    print(f"\nRefining {folder}")
    output_path = os.path.join(output_folder, Path(folder).name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    files = glob.glob(folder + "**/**", recursive=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG.workers) as executor:
        futures = []
        for file in files:
            if os.path.isfile(file):
                futures.append(executor.submit(refine_file, output_path, file))

        with tqdm(total=len(futures)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                f = future.result()
                pbar.update(1)

    # reverse look up, remove refined files that are no longer in the original folder
    refined_files = glob.glob(output_path + "**/**", recursive=True)
    print("===Removing files that are no longer in the original folder")
    for refined_file in refined_files:
        if os.path.isfile(refined_file) and refined_file not in OUTPUT_FILES:
            print(f"Removing {refined_file}")
            os.remove(refined_file)


def refine_documents(language, folder_path, output_path, server_url, api_key):
    global CONFIG
    with open("./impl/configs/refine_config.json", "r", encoding="utf-8") as f:
        CONFIG = Config.parse_obj(json.load(f)[language])
    if server_url is not None:
        CONFIG.servers = [server_url]
    if api_key is not None:
        CONFIG.api_key = api_key

    custom_data_folder = folder_path
    output_folder = output_path

    refine(custom_data_folder, output_folder)

    print(f"Completed total {len(OUTPUT_FILES)} files!")
    return  {len(OUTPUT_FILES)}
