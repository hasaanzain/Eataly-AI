import streamlit as st
from langsmith import traceable
from pathlib import Path
from datetime import datetime, timezone
import gspread
from google.oauth2.service_account import Credentials

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


@st.cache_resource
def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )

    client = gspread.authorize(creds)

    sheet_id = st.secrets["GOOGLE_SHEET_ID"]
    return client.open_by_key(sheet_id).sheet1


@traceable(name="User Query")
def track_user_query(user_query):
    timestamp = datetime.now(timezone.utc).isoformat()

    sheet = get_sheet()
    sheet.append_row([timestamp, user_query])

    return {
        "timestamp_utc": timestamp,
        "user_query": user_query
    }


vectordb = load_vectordb()
llm = load_llm()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Ciao! Welcome to Eataly! How can I help you today?"
        }
    ]


if "user_queries" not in st.session_state:
    st.session_state.user_queries = []


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask me about menus, HR policies, wines, procedures...")


if prompt:
    user_query = prompt

    st.session_state.user_queries.append(user_query)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.write_stream(
                    chatbot_stream(
                        user_query,
                        vectordb=vectordb,
                        k=4,
                        llm=llm
                    )
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                # Log the query only after the response has streamed to the user.
                # This avoids making the user wait for Google Sheets logging.
                try:
                    track_user_query(user_query)
                except Exception as log_error:
                    print(f"Query logging failed: {log_error}")

            except Exception as e:
                # Full details go to the logs (CloudWatch in AWS); users get a generic message.
                print(f"Chat request failed: {e!r}")
                error_msg = "Sorry, something went wrong. Please try again or ask a manager."
                st.error(error_msg)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_msg
                    }
                )
