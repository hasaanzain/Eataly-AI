# 🍝 Eataly AI

Internal Retrieval-Augmented Generation (RAG) Assistant for Eataly Staff

Eataly AI is a Streamlit-based internal chatbot that helps staff access operational, HR, menu, wine, and procedural information from internal PDF documentation. The system uses a Retrieval-Augmented Generation pipeline built with LangChain, Chroma, and OpenAI models.

---

## 🚀 Features

* PDF knowledge base ingestion
* Semantic search using OpenAI embeddings
* GPT-powered contextual responses
* Real-time streaming responses
* Session-based chat history
* Built-in guardrails:

  * No employee personal data disclosure
  * No order placement or availability checks
  * Escalation for out-of-scope queries
  * Italian word translation support

---

## 🏗️ Architecture

1. Load PDFs from `eataly_ai_knowledge_base/`
2. Split into overlapping chunks (512 size, 50 overlap)
3. Generate embeddings with `text-embedding-3-small`
4. Store in Chroma vector database
5. Retrieve top-k relevant chunks
6. Generate response with `gpt-5-nano`
7. Stream output in Streamlit UI

---

## 📂 Project Structure

```
.
├── chatbot.py
├── streamlit.py
├── eataly_ai_knowledge_base/
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```
git clone https://github.com/yourusername/eataly-ai.git
cd eataly-ai
```

### 2. Create Virtual Environment

```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

---

## ▶️ Run the App

```
streamlit run streamlit.py
```

App runs at:

```
http://localhost:8501
```

---

## 🛠 Configuration

* Adjust chunking in `get_chunks()`
* Modify retrieval depth with `k` in `chatbot_stream()`
* Change model in `get_llm()`

---

## 🛡️ Safety Design

The assistant is restricted to staff use and enforces:

* Context-grounded responses only
* Manager escalation for unknown answers
* Privacy protection
* No operational authority

---

## 📈 Observability

LangSmith tracing is enabled via `@traceable` decorators for debugging and monitoring.

---

## 📜 License

Internal use only.
