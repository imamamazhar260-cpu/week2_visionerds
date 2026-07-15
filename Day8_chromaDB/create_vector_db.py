from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("data/week 12-13.pdf")
documents = loader.load()

print("Total Pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="imama",
    persist_directory="chroma_db"
)

print("Vector database created successfully!")


embeddings = embedding_model.embed_documents(
    [doc.page_content for doc in chunks]
)

print("\nEmbedding Dimension:", len(embeddings[0]))

print(embeddings)