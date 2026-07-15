from langchain_text_splitters import RecursiveCharacterTextSplitter
with open("sample.txt", "r", encoding="utf-8") as file:
    text = file.read()
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
    chunks = splitter.split_text(text)
    print("Total Chunks:", len(chunks))
    for i, chunk in enumerate(chunks[:5]):
       print(f"\nChunk {i+1}\n")
       print(chunk)
       print("-" * 60)