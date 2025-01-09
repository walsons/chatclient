import streamlit as st
from pathlib import Path
import os
from impl.generate_oemchat_kbs import EmbeddingService
from impl.start_server import start_server, SeverType


# @st.cache_data
def generate_knowledge_base(folder_path, add_refined_documents=False, server_url=None, api_key=None, gkb_method_choice="Local"):
    try:
        if folder_path is None or folder_path == "":
            st.warning("⚠️Please enter the folder path of the documents")
            return
        if add_refined_documents and not os.path.exists(os.path.join(Path(folder_path).parent.absolute(), "refined_output")):
            st.warning("⚠️Not found refined documents, please refine the documents first")
            return
        if gkb_method_choice == "Local":
            start_server(server_type=SeverType.EMBEDDING)
        service = EmbeddingService(folder_path, add_refined_documents, server_url, api_key)
        service.start()
        st.info(f"✅Knowledge base generated, output files are saved in {service.kb_output_path}")
    except Exception as e:
        st.error(f"Error: {e}")


folder_path = st.text_input("Enter the folder path of the documents")
add_refined_documents = st.checkbox("Add refined documents")

server_url = None
api_key = None

gkb_method_choice = st.radio("Generate knowledge base in local or on server", ["Local", "Server"])

if gkb_method_choice == "Server":
    st.info("Note: The embedding server must using \"embedding model name\"", icon="ℹ️")
    server_url = st.text_input("Server URL (e.g. http://127.0.0.1:1234/v1)")
    api_key = st.text_input("API Key")

button = st.button(
    label="Generate",
    on_click=generate_knowledge_base,
    args=[folder_path, add_refined_documents, server_url, api_key, gkb_method_choice],
)
