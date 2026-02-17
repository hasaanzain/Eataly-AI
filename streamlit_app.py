import streamlit as st
from langsmith import traceable
from pathlib import Path

from chatbot import chatbot_stream, get_vectordb, get_llm

st.set_page_config(page_title="Eataly AI")
st.title("Eataly AI")

path = Path("eataly_ai_knowledge_base")

@st.cache_resource
def load_vectordb():
    return get_vectordb(path)

@st.cache_resource
def load_llm():
    return get_llm()

vectordb = load_vectordb()
llm = load_llm()

# Initialize chat history with assistant greeting
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Ciao! Welcome to Eataly! How can I help you today?"}
    ]

# Display existing messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

@traceable
def get_query():
    return st.chat_input("Ask me about menus, HR policies, wines, procedures...")

prompt = get_query()


if prompt:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.write_stream(chatbot_stream(prompt, vectordb=vectordb, k=4, llm=llm))
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
