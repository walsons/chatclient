import streamlit as st
import subprocess
import enum


class SeverType(enum.Enum):
    UNKNOWN = 0,
    CHAT = 1,
    REFINE = 2,
    EMBEDDING = 3,

def start_local_llama_server(args_list):
    llama_server = subprocess.Popen(
        args_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    ready_text = "starting the main loop"
    output = ""
    for line in iter(llama_server.stdout.readline, ''):
        output += line
        print(line, end='')
        if ready_text in output:
            break
    print("============== local llama server started ================")
    st.session_state.running_server = llama_server


def start_server(server_type):
    if "running_server_type" not in st.session_state:
        st.session_state.running_server_type = SeverType.UNKNOWN
    if "running_server" not in st.session_state :
        st.session_state.running_server = None

    if server_type != st.session_state.running_server_type:
        if st.session_state.running_server is not None:
            st.session_state.running_server.terminate()
        
        if server_type == SeverType.CHAT:
            # current implementation is using local llama server, will change to AMD chat server later
            start_local_llama_server(["bin/llama-server.exe", "-m", "bin/meta-llama-3.1-8b-instruct.Q4_K_M.gguf", "--port", "8081"])
        elif server_type == server_type.REFINE:
            start_local_llama_server(["bin/llama-server.exe", "-m", "bin/meta-llama-3.1-8b-instruct.Q4_K_M.gguf", "--port", "8082"])
        elif server_type == server_type.EMBEDDING:
            start_local_llama_server(["bin/llama-server.exe", "-m", "bin/bge-large-en-v1.5-q8_0.gguf", "--embeddings", "-c", "4096", "--port", "8083"])
        else:
            raise ValueError("Invalid server type")

        st.session_state.running_server_type = server_type
