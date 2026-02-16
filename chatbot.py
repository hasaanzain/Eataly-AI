# chatbot.py

from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def build_vectordb_from_folder(folder_path: str):
    folder = Path(folder_path)

    file_names = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() == ".pdf"]

    loaded_docs = []
    for file in file_names:
        loader = PyPDFLoader(str(file))
        loaded_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "  ", " ", ""],
    )

    chunk_docs = splitter.split_documents(loaded_docs)
    ids = [str(i) for i in range(len(chunk_docs))]

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    vectordb = Chroma.from_documents(
        documents=chunk_docs,
        embedding=embedding_model,
        ids=ids,
    )
    return vectordb

def get_llm():
    return ChatOpenAI(model="gpt-5-nano", temperature=0.6)

def chatbot(query, vectordb, k=4, llm=None):
    if llm is None:
        llm = get_llm()

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_template("""
You are a friendly and helpful bot designed to help out the workers of the italian restaurant Eataly.
You are only responding to staff and not any customer inquiries.
Use full english sentences to answer.
If you do not understand a query, ask 1 follow up question asking them to clarify.
If you are not able to answer their question with the provided context, say "Unfortunately this is beyond the scope of my knowledge. For the most updated answer, please ask a manager."
If the user brings up an italian word, translate and explain the meaning of the italian word.
Do not reveal any personal information of any employee under any circumstance.
You cannot place any orders or check availability for dishes.
CONTEXT: {context}
QUESTION: {question}
Answer:
""")

    formatted_prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(formatted_prompt)
    return response.content
