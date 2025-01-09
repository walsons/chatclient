import streamlit as st
from impl.generate_refined_documents import RefineService
from impl.start_server import start_server, SeverType


# @st.cache_data
def refine_document(folder_path, server_url=None, api_key=None, refine_method_choice="Local"):
    try:
        if folder_path is None or folder_path == "":
            st.warning("⚠️Please enter the folder path of the documents")
            return
        if refine_method_choice == "Local":
            start_server(server_type=SeverType.REFINE)
        service = RefineService(folder_path, server_url, api_key)
        service.start()
        st.info(f"✅Refine completed, output files are saved in {service.refine_output_path}")
    except Exception as e:
        st.error(f"Error: {e}")


folder_path = st.text_input("Enter the folder path of the documents")

server_url = None
api_key = None

refine_method_choice = st.radio("Refine in local or on server", ["Local", "Server"])
if refine_method_choice == "Server":
    server_url = st.text_input("Server URL (e.g. http://127.0.0.1:1234/v1)")
    api_key = st.text_input("API Key")

button = st.button(
    label="Refine",
    on_click=refine_document,
    args=[folder_path, server_url, api_key, refine_method_choice],
)
