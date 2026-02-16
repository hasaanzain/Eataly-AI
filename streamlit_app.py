import streamlit as st
from pathlib import Path

from chatbot import build_vectordb_from_folder, get_llm, chatbot_stream

st.set_page_config(page_title="Eataly AI")
st.title("Eataly AI")

BASE_DIR = Path(__file__).resolve().parent
KB_FOLDER = str(BASE_DIR / "eataly_ai_knowledge_base")


@st.cache_resource
def load_vectordb():
    print("BUILDING VECTORDB NOW")
    return build_vectordb_from_folder(KB_FOLDER)


@st.cache_resource
def load_llm():
    return get_llm()


# Load heavy resources once
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

prompt = st.chat_input("Ask about menus, HR policies, wines, procedures...")

if prompt:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # full = chatbot_stream()
            full = st.write_stream(chatbot_stream(prompt, vectordb=vectordb, k=4, llm=llm))

        st.session_state.chat_history.append({"role": "assistant", "content": full})

