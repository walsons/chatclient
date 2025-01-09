# -------------------------------------------------------------------------
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
# -------------------------------------------------------------------------
import os
from pathlib import Path
from impl.scripts.create_oem_kb import (
    generate_kb,
    join_and_make_path,
    VECTOR_STORE_PATH,
    DB_FILE_NAME,
)


CHAT_REGION_GLOBAL = "en-US"
CHAT_REGION_CHINA = "zh-CN"

class EmbeddingService:
    def __init__(self, folder_path, add_refined_documents, server_url, api_key):
        self.languages = [CHAT_REGION_GLOBAL]
        self.base_path = Path(folder_path).parent.absolute()
        self.document_path = Path(folder_path).absolute()
        # self.kb_output_path = os.path.join(self.base_path, "kb_output")
        self.kb_output_path = join_and_make_path(VECTOR_STORE_PATH, DB_FILE_NAME)
        self.kb_output_path = os.path.abspath(self.kb_output_path)  # just for replace '/' with '\'
        self.refine_output_path = os.path.join(self.base_path, "refined_output") if add_refined_documents else None
        self.server_url = server_url
        self.api_key = api_key

    def start(self):
        generate_kb(self.languages[0], self.document_path, self.refine_output_path, self.kb_output_path, self.server_url, self.api_key)
        