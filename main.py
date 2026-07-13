from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading model...")


model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!\n")


sentences = [
    "I love pizza.",
    "Pizza tastes delicious.",
    "Cats are wonderful pets.",
    "Dogs are loyal animals.",
    "Artificial Intelligence is amazing.",
    "Machine Learning is part of AI."
]


embeddings = model.encode(sentences)
print("Embedding Shape:", embeddings.shape)
print()


for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("First 10 values:", embedding[:10])
    print("-" * 50)

# Compute cosine similarity
similarity_matrix = cosine_similarity(embeddings)

print("\n========== SIMILARITY MATRIX ==========\n")
print(similarity_matrix)

print("\n========== SIMILARITY SCORES ==========\n")


for i in range(len(sentences)):
    for j in range(len(sentences)):
        print(f"Sentence {i+1}: {sentences[i]}")
        print(f"Sentence {j+1}: {sentences[j]}")
        print(f"Similarity: {similarity_matrix[i][j]:.3f}")
        print("-" * 60)


def most_similar(query, sentences):
    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    best_index = np.argmax(scores)

    return sentences[best_index], scores[best_index]


query = "AI is changing the world."

best_sentence, score = most_similar(query, sentences)

print("\n========== MOST SIMILAR RESULT ==========\n")
print("Query:", query)
print("Most Similar Sentence:", best_sentence)
print("Similarity Score:", round(score, 3))