import os

from dotenv import load_dotenv
from groq import Groq

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


vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


data = vector_db.get()

texts = data["documents"]
metadatas = data["metadatas"]

print("Database Loaded Successfully")

tokenized_texts = [text.split() for text in texts]

bm25 = BM25Okapi(tokenized_texts)

print("Hybrid Retriever Ready")


def dense_search(query, k=3):
    results = vector_db.similarity_search_with_score(
        query,
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

    results = []

    for i in ranked_indices[:k]:

        results.append(
            Document(
                page_content=texts[i],
                metadata=metadatas[i]
            )
        )

    return results


def hybrid_search(query, k=5):

    dense_results = dense_search(query, k=3)
    keyword_results = keyword_search(query, k=3)

    final_docs = []
    seen = set()

    for doc, score in dense_results:

        if doc.page_content not in seen:
            final_docs.append(doc)
            seen.add(doc.page_content)

    for doc in keyword_results:

        if doc.page_content not in seen:
            final_docs.append(doc)
            seen.add(doc.page_content)

    return final_docs[:k]


while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit", "bye"]:

        print("\nGoodbye!")
        break

    retrieved_docs = hybrid_search(question)

    context = ""

    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not explicitly present in the context,
reply exactly:

I don't know.

Do not use outside knowledge.
Do not guess.

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

    for i, doc in enumerate(retrieved_docs, start=1):

        page = doc.metadata.get("page", "Unknown")

        print(f"\nChunk {i}")
        print(f"Page: {page}")
        print("-" * 60)
        print(doc.page_content[:300] + "...")