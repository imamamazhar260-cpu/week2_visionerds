

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "week 12-13.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()


print("Total Pages:", len(documents))


text = ""

for page in documents:
    text += page.page_content


print("Total Characters:", len(text))

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
for i, chunk in enumerate(chunks[:5]):

    print("\n==========================")
    print("Chunk:", i+1)
    print("==========================")

    print(chunk.page_content)

