# chatbot.py

from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


from pathlib import Path
path = Path("eataly_ai_knowledge_base")


def get_doc_names(path):
    file_names = []         
    for file in path.rglob("*"):    
        if file.is_file():        
            if file.suffix.lower() == ".pdf":   
                file_names.append(file)        
    return file_names          


def load_docs(file):
    loader = PyPDFLoader(str(file))
    return loader.load() 



def get_text(file_names):
    corpus = []
    for file in file_names:
        corpus.extend(load_docs(file))
    return corpus



def get_chunks(corpus):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "  ", " ", ""],
    )
    return splitter.split_documents(corpus)



def build_vectordb(chunks):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    ids = [str(i) for i in range(0, len(chunks))]
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        ids=ids
    )
    return vectordb



def get_llm():
    # streaming=True enables llm.stream(...)
    return ChatOpenAI(model="gpt-5-nano", temperature=0.6, streaming=True)


def get_vectordb(path):
    file_names = get_doc_names(path)
    corpus = get_text(file_names)
    chunks = get_chunks(corpus)
    vectordb = build_vectordb(chunks)
    return vectordb


def chatbot_stream(query, vectordb=None, k=3, llm=None):
    if llm is None:
        llm = get_llm()

    if vectordb is None:
        vectordb = get_vectordb(path)

    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt_template = ChatPromptTemplate.from_template(
        """
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
"""
    )

    chain = prompt_template | llm | StrOutputParser()
    return chain.stream({"context": context, "question": query})



