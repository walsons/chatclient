# -------------------------------------------------------------------------
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
# -------------------------------------------------------------------------
import os
from pathlib import Path
from impl.scripts.refine_doc import refine_documents


CHAT_REGION_GLOBAL = "en-US"
CHAT_REGION_CHINA = "zh-CN"


class RefineService:
    def __init__(self, folder_path, server_url, api_key):
        self.languages = [CHAT_REGION_GLOBAL]
        self.base_path = Path(folder_path).parent.absolute()
        self.doc_folder = Path(folder_path).name
        self.refine_output_path = []
        self.server_url = server_url
        self.api_key = api_key

    def start(self):
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    refine_documents,
                    self.languages[0],
                    os.path.join(self.base_path, self.doc_folder).replace(
                        "\\", "/"
                    ),
                    os.path.join(
                        self.base_path, "refined_output"
                    ).replace("\\", "/"),
                    self.server_url,
                    self.api_key,
                )
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    self.refine_output_path = [os.path.join(
                        self.base_path, "refined_output"
                    ).replace("\\", "/")]

                except Exception as exc:
                    print(f"Generated an exception: {exc}")

                else:
                    print(f"Refined {res} files")


if __name__ == "__main__":
    service = RefineService()
    service.start()
