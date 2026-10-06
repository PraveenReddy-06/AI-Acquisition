import chromadb
import ollama
from pypdf import PdfReader
from pathlib import Path


# --------------------------------
# 1. Documents to ingest
# --------------------------------

documents_to_ingest = [
    "pypdf/notes.pdf",
    "pypdf/py-essentials-2.pdf",
    "pypdf/KlInnovationHub_ER_Diagram_A4.pdf"
]


# --------------------------------
# 2. Create persistent ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_or_create_collection(
    name="pdf_documents"
)


# --------------------------------
# 3. Process every PDF
# --------------------------------

for file_path in documents_to_ingest:

    print("\nProcessing:", file_path)
    reader = PdfReader(file_path)
    source = Path(file_path).name

    chunk_size = 100
    overlap = 20
    chunk_number = 0  
    chunks = []  


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

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
                "source": source,
                "page": page_number,
                "chunk": chunk_number
            })

            chunk_number += 1

            start = start + chunk_size - overlap


    print("Chunks:", len(chunks))


    # --------------------------------
    # 5. Create embeddings
    # --------------------------------

    embeddings = []

    for chunk in chunks:

        response = ollama.embed(
            model="nomic-embed-text",
            input=chunk["text"]
        )

        embeddings.append(
            response["embeddings"][0]
        )


    # --------------------------------
    # 6. Prepare ChromaDB data
    # --------------------------------

    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:

        # Include source in ID
        ids.append(
            f"{chunk['source']}_{chunk['chunk']}"
        )

        texts.append(
            chunk["text"]
        )

        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk": chunk["chunk"]
        })


    # --------------------------------
    # 7. Store in ChromaDB
    # --------------------------------

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )


    print(
        f"Stored {len(chunks)} chunks from {source}"
    )


print("\nTotal chunks in database:")
print(collection.count())