from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing Chroma DB
vector_db = Chroma(
    collection_name="imama",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

query = input("Enter your question: ")

# Query embedding
query_embedding = embedding_model.embed_query(query)

print("\n==============================")
print("QUERY VECTOR")
print("==============================")
print("Dimension:", len(query_embedding))
print("First 20 values:")
print(query_embedding[:20])


results = vector_db.similarity_search_with_score(
    query,
    k=3
)

print("\n==============================")
print("TOP RESULTS")
print("==============================")

for i, (doc, score) in enumerate(results, 1):

    print(f"\nResult {i}")
    print("-" * 60)

    print("Similarity Score:", score)

    
    doc_embedding = embedding_model.embed_query(doc.page_content)

    print("Embedding Dimension:", len(doc_embedding))
    print("First 20 Numbers:")
    print(doc_embedding[:20])

    print("\nPage:", doc.metadata["page"] + 1)
    print("\nDocument Text:")
    print(doc.page_content)

print("\n==============================")
print("Stored Embeddings in Chroma")
print("==============================")

collection = vector_db._collection

data = collection.get(include=["embeddings", "documents"])

print("Total Stored Vectors:", len(data["embeddings"]))

print("\nFirst Stored Vector (First 20 Numbers):")
print(data["embeddings"][0][:20])