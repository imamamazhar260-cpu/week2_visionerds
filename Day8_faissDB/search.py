from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

vector_db = FAISS.load_local(
    "faiss_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")


query = input("Enter your question: ")


query_embedding = embedding_model.embed_query(query)

print("\n========== QUERY EMBEDDING ==========")
print("Embedding Dimension:", len(query_embedding))
print("First 20 Values:")
print(query_embedding[:20])


results = vector_db.similarity_search_with_score(
    query,
    k=3
)

print("\n========== SEARCH RESULTS ==========\n")

for i, (doc, score) in enumerate(results, start=1):

    print("=" * 70)
    print(f"Result {i}")
    print("=" * 70)

    print("Similarity Score:", score)

    print("\nDocument Embedding")
    doc_embedding = embedding_model.embed_query(doc.page_content)
    print("Dimension:", len(doc_embedding))
    print("First 20 Values:")
    print(doc_embedding[:20])

    print("\nPage Number:", doc.metadata["page"] + 1)

    print("\nRetrieved Text:\n")
    print(doc.page_content)

    print("\n")