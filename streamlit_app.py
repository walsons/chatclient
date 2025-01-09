import streamlit as st


pg = st.navigation([
    st.Page(page="subpages/chatbot.py", title="Chatbot", icon="💬"),
    st.Page(page="subpages/refine_document.py", title="Refine Document", icon="📝"),
    st.Page(page="subpages/generate_knowledge_base.py", title="Generate Knowledge Base", icon="🧠"),
])

pg.run()

# st.title("Hello World")