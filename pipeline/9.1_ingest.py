import chromadb
import ollama
from pypdf import PdfReader


# -----------------------------
# 1. Read PDF
# -----------------------------

reader = PdfReader("pypdf/notes.pdf")
chunks = []
chunk_size = 100
overlap = 20
chunk_number = 0

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    if not page_text:
        continue
    words = page_text.split()

    start = 0
    while start < len(words):
        chunk_text = " ".join(
            words[start:start + chunk_size]
        )
        chunks.append({
            "text": chunk_text,
            "page": page_number,
            "chunk": chunk_number
        })
        chunk_number += 1
        start = start + chunk_size - overlap


print("Number of chunks:", len(chunks))


# -----------------------------
# 2. Create embeddings
# -----------------------------

embeddings = []
for chunk in chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk["text"]
    )
    embeddings.append(
        response["embeddings"][0]
    )
print("Embeddings created:", len(embeddings))


# -----------------------------
# 3. Persistent ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_or_create_collection(
    name="pdf_documents"
)


# -----------------------------
# 4. Prepare IDs + metadata
# -----------------------------

ids = []
metadatas = []
documents = []

for chunk in chunks:
    ids.append(
        f"chunk_{chunk['chunk']}"
    )
    documents.append(
        chunk["text"]
    )
    metadatas.append({
        "page": chunk["page"],
        "chunk": chunk["chunk"]
    })


# -----------------------------
# 5. Store in ChromaDB
# -----------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)


print("Documents stored in ChromaDB.")
print("Total documents:", collection.count())