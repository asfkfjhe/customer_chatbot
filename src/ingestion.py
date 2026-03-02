from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import os

DATA_PATH = "data/knowledge_base"
DB_PATH = "vector_store/chroma_db"


def build_vector_store():

    loader = DirectoryLoader(DATA_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large" 
    )

    vector_db = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=DB_PATH
    )

    vector_db.persist()

    print("✅ Vector database built successfully")

    
if __name__ == "__main__":
    build_vector_store()