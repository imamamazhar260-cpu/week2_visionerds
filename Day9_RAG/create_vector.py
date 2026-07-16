import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

loader = PyPDFLoader("Data/week 12-13.pdf")
documents = loader.load()

print(f"Total Pages: {len(documents)}")

text_splitter = SemanticChunker(
    embeddings=embedding_model
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")
    print("Old Chroma Database Deleted")

Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("New Chroma Database Created Successfully")