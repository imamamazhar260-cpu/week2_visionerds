from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

print("Embedding model loaded successfully!")

texts = [doc.page_content for doc in chunks]
embeddings = embedding_model.embed_documents(texts)

print("\n========== EMBEDDINGS ==========")
print("Total Embeddings:", len(embeddings))
print("Embedding Dimension:", len(embeddings[0]))

print("\nFirst Embedding (First 20 Values):")
print(embeddings[0][:20])

print("\nSecond Embedding (First 20 Values):")
print(embeddings[1][:20])

vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

vector_db.save_local("faiss_db")

print("\nFAISS database created successfully!")