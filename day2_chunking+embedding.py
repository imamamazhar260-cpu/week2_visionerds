from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import json


# -----------------------------
# Load PDF
# -----------------------------

pdf_path = "week 12-13.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Total Pages:", len(documents))


# -----------------------------
# Chunking
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


chunks = text_splitter.split_documents(documents)

print("Total Chunks:", len(chunks))


# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# -----------------------------
# Create embeddings
# -----------------------------

texts = []

for chunk in chunks:
    texts.append(chunk.page_content)


embeddings = model.encode(texts)

print("Embedding completed")


# -----------------------------
# Save chunk + metadata + embedding
# -----------------------------

data = []


for i, chunk in enumerate(chunks):

    item = {

        "chunk_id": i,

        "text": chunk.page_content,

        "metadata": {
            "page": chunk.metadata["page"] + 1,
            "source": chunk.metadata["source"]
        },

        "embedding": embeddings[i].tolist()

    }


    data.append(item)



# -----------------------------
# Save JSON file
# -----------------------------

with open(
    "embedded_chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        data,
        file,
        indent=4,
        ensure_ascii=False
    )


print("JSON file created successfully")