import os

from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rank_bm25 import BM25Okapi


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

DB_PATH = "chroma_db"
PDF_PATH = "Data/week 12-13.pdf"


if os.path.exists(DB_PATH):

    print("Loading Existing ChromaDB...")

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

else:

    print("ChromaDB not found.")
    print("Creating Vector Database...\n")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"Pages Loaded : {len(documents)}")

    text_splitter = SemanticChunker(
        embeddings=embedding_model
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Chunks Created : {len(chunks)}")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print("Vector Database Created Successfully")


data = vector_db.get()

texts = data["documents"]

metadatas = data["metadatas"]

tokenized_texts = [
    text.split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_texts)

print("\nHybrid Retriever Ready")
def dense_search(query, k=3):

    results = vector_db.similarity_search_with_score(
        query=query,
        k=k
    )

    return results


def keyword_search(query, k=3):

    scores = bm25.get_scores(query.split())

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    documents = []

    for i in ranked_indices[:k]:

        documents.append(
            Document(
                page_content=texts[i],
                metadata=metadatas[i]
            )
        )

    return documents


def hybrid_search(query, k=5):

    dense_results = dense_search(query, k=3)
    keyword_results = keyword_search(query, k=3)

    final_results = []
    seen = set()

    for doc, score in dense_results:

        if doc.page_content not in seen:

            final_results.append(
                {
                    "document": doc,
                    "score": score,
                    "source": "Dense"
                }
            )

            seen.add(doc.page_content)

    for doc in keyword_results:

        if doc.page_content not in seen:

            final_results.append(
                {
                    "document": doc,
                    "score": None,
                    "source": "BM25"
                }
            )

            seen.add(doc.page_content)

    return final_results[:k]


print("\nRAG Chatbot Ready")
print("Type 'exit' to quit.\n")
while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit", "bye"]:
        print("\nGoodbye!")
        break

    retrieved_docs = hybrid_search(question)

    context = ""

    for item in retrieved_docs:
        context += item["document"].page_content
        context += "\n\n"

    prompt = f"""
You are a helpful AI Assistant.

Answer ONLY using the context below.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the answer is not present in the context, reply exactly:
I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    if "i don't know" in answer.lower():
        continue

    print("\n" + "=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for i, item in enumerate(retrieved_docs, start=1):

        doc = item["document"]

        print(f"\nChunk {i}")

        if item["score"] is not None:
            print(f"Similarity Score : {item['score']:.4f}")

        print(f"Page : {doc.metadata.get('page')}")

        print("-" * 80)
        print(doc.page_content)
        print("-" * 80)