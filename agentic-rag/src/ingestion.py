from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import os

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, 
    chunk_overlap=10
)
doc_splits = text_splitter.split_documents(docs_list)

def _get_vectorstore():
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        persist_directory="./.chroma",
        embedding=AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    )
    return vectorstore

retriever = None
vectorstore = None

def get_retriever():
    global retriever
    global vectorstore
    vectorstore = _get_vectorstore()
    if retriever is None:
        retriever = vectorstore.as_retriever()
    return retriever
