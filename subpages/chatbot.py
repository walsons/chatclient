import streamlit as st
from openai import OpenAI
import time
import os
import json
import requests
from impl.start_server import start_server, SeverType


chatbot_label = "assistant"
user_label = "user"
chatbot_avatar = "assets/chatbot.png"
user_avatar = "assets/user.png"


class Chatbot:
    def __init__(self):
        self.chatbot_options = ["AMD Chatbot", "OEM Chatbot"]
        self.chatbot_options_choice = None
        # Read chat history from the json files in the chat_history folder and display first 30 characters
        self.chat_historys = []
        if os.path.exists("chat_history"):
            for file in os.listdir("chat_history"):
                with open(os.path.join("chat_history", file), "r") as f:
                    self.chat_historys.append(json.load(f))
        self.llama_server_url = "http://127.0.0.1:8081"
        self.amdchat_server_url = "http://127.0.0.1:40000/v1"
        self.llama_client = OpenAI(base_url=self.llama_server_url, api_key="local_llm")
        self.amdchat_client = OpenAI(base_url=self.amdchat_server_url, api_key="local_llm")
        self.greet_sentence = "Hello! I am a chatbot. How can I help you today?"
        self.stream = True
    
    def __show_chat_options(self):
        choice = st.radio(r"$\textsf{\normalsize Chat Options}$", self.chatbot_options, horizontal=True)
        if choice == "OEM Chatbot":
            self.chatbot_options_choice = "oem"
        else:
            self.chatbot_options_choice = "amd"

    def __show_chat_history(self):
        st.sidebar.markdown(r"$\textsf{\normalsize Chat History}$")
        for index, history in enumerate(self.chat_historys):
            label = f"{index + 1}\\. {history[0]['content'][:30]}..."  # no idea why double backslash is needed
            st.sidebar.button(label=label, use_container_width=True)
    
    def __show_chat_session(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
            st.session_state.chat_messages.append({"role": chatbot_label, "avatar": chatbot_avatar, "content": self.greet_sentence})

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

        user_input = st.chat_input("ask me something")
        if user_input:
            if user_input.lower() == "clear":
                self.__new_session()
                return
            with st.chat_message(user_label, avatar=user_avatar):
                st.markdown(user_input)
            st.session_state.chat_messages.append({"role": user_label, "avatar": user_avatar, "content": user_input})

            with st.chat_message(chatbot_label, avatar=chatbot_avatar):
                if self.stream:
                    chatbot_response = st.write_stream(self.__response_generator(user_input, True))
                else:
                    chatbot_response = self.__response_generator(user_input, True)
                    st.markdown(chatbot_response)
            st.session_state.chat_messages.append({"role": chatbot_label, "avatar": chatbot_avatar, "content": chatbot_response})
    
    def __new_session(self):
        self.__save_chat_history()
        st.session_state.chat_messages = []
        st.rerun()

    def __save_chat_history(self):
        if "chat_messages" in st.session_state and len(st.session_state.chat_messages) > 1:
            # Write chat messages to a csv file named with timestamp
            if not os.path.exists("chat_history"):
                os.makedirs("chat_history")
            with open("chat_history/chat_history_" + str(time.time_ns()) + ".json", "w") as f:
                # Skip the first message if it is the greet sentence
                if (st.session_state.chat_messages[0]["role"] == "chatbot" and st.session_state.chat_messages[0]["content"] == self.greet_sentence):
                    json.dump(st.session_state.chat_messages[1:], f)
                    self.chat_historys.append(st.session_state.chat_messages[1:])
                else:
                    json.dump(st.session_state.chat_messages, f)
                    self.chat_historys.append(st.session_state.chat_messages)
    
    def __response_generator(self, user_input, is_use_amdchat=False):
        if is_use_amdchat:
            return self.__amdchat_response_generator(user_input)
        else:
            return self.__llama_response_generator(user_input)

    def __llama_response_generator(self, user_input):
        start_server(SeverType.CHAT)
        completion = self.llama_client.chat.completions.create(
            model = "local",
            messages = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages
            ],
            stream=self.stream
        )
        if self.stream:
            return self.__response_content_stream(completion)
        return self.__response_content(completion)

    def __amdchat_response_generator(self, user_input):
        # TODO: start_server(SeverType.CHAT)
        completion = self.amdchat_client.chat.completions.create(
            model = "local",
            messages = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages
            ],
            stream=self.stream,
            extra_body={"topic_id": self.chatbot_options_choice},
        )
        if self.stream:
            return self.__response_content_stream(completion)
        return self.__response_content(completion)

    def __response_content(self, completion):
        return completion.choices[0].message.content

    def __response_content_stream(self, completion):
        for chunk in completion:
            if chunk.choices[0].delta.content is None:
                break
            yield chunk.choices[0].delta.content
    
    def __amdchat_response_generator_simple(self, user_input):
        # TODO: start_server(SeverType.CHAT)
        amdchat_server_url = "http://127.0.0.1:40000"
        simple_chat_endpoint = f"{amdchat_server_url}/v1/chat/simple"
        # First, get the chat_id
        message = {"chat_id": "", "message": user_input}
        response = requests.post(simple_chat_endpoint, json=message, timeout=30)
        while response.status_code == 503:
            print("Server is busy, retrying...")
            response = requests.post(self.url, json=message, timeout=30)
        response = response.json()
        chat_id, error = response["chat_id"], response["error"]
        # Second, get the response
        if error == 0:
            get_params = {"chat_id": chat_id}
            completed = False
            while not completed:
                response = requests.get(simple_chat_endpoint, params=get_params, timeout=30)
                response = response.json()
                request_message_id = response["request_message_id"]
                response_message_id = response["response_message_id"]
                message = response["message"]
                completed = response["completed"]
                generation_exception = response["generation_exception"]
                progress = response["progress"]
                if message != "":
                    yield message

    def one_loop(self):
        self.__show_chat_options()
        self.__show_chat_history()
        self.__show_chat_session()


if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()
st.session_state.chatbot.one_loop()


# Questions examples to ask the chatbot:
# What are the key factors that contribute to a successful career in the field of data science?
# Can you suggest some effective strategies for maintaining a balanced diet and exercise routine?
# How can I improve my public speaking skills to excel in professional presentations?