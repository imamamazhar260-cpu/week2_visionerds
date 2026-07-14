import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open(
    "embedded_chunks.json",
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)



print("Chunks loaded:", len(data))

model = SentenceTransformer("all-MiniLM-L6-v2")



query = "Explain indexes and their advantages"




query_embedding = model.encode([query])


best_score = -1
best_chunk = None


for item in data:


    chunk_vector = np.array(
        item["embedding"]
    )


    score = cosine_similarity(
        query_embedding,
        [chunk_vector]
    )[0][0]



    if score > best_score:

        best_score = score

        best_chunk = item







print("\n========== RESULT ==========")


print("\nSimilarity Score:")
print(best_score)


print("\nChunk ID:")
print(best_chunk["chunk_id"])


print("\nPage Number:")
print(best_chunk["metadata"]["page"])


print("\nPDF:")
print(best_chunk["metadata"]["source"])


print("\nAnswer Text:")
print(best_chunk["text"])