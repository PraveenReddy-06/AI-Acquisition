import chromadb
import ollama


# -----------------------------
# 1. Connect to existing DB
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_data"
)
collection = client.get_collection(
    name="pdf_documents"
)
print("Documents in database:", collection.count())


# -----------------------------
# 2. Ask question
# -----------------------------

question = input("\nAsk a question: ")


# -----------------------------
# 3. Create question embedding
# -----------------------------

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]
print("question embedding:",question_embedding)

# -----------------------------
# 4. Search ChromaDB
# -----------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# -----------------------------
# 5. Display retrieved chunks
# -----------------------------

print("\nRetrieved chunks:\n")
for i in range(len(results["documents"][0])):
    print("Rank:", i + 1)
    print(
        "Distance:",
        results["distances"][0][i]
    )
    print(
        "Page:",
        results["metadatas"][0][i]["page"]
    )
    print(
        "Chunk:",
        results["metadatas"][0][i]["chunk"]
    )
    print(
        "Text:",
        results["documents"][0][i]
    )
    print("-" * 60)


# -----------------------------
# 6. Build context
# -----------------------------

context = "\n\n".join(
    results["documents"][0]
)


# -----------------------------
# 7. Send context to Qwen
# -----------------------------

prompt = f"""
Answer the question using ONLY the context below.

If the answer is not present in the context,
say: "I don't know based on the provided document."

Do not use outside knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


response = ollama.generate(
    model="qwen2:0.5b",
    prompt=prompt
)


# -----------------------------
# 8. Final answer
# -----------------------------

print("\nAnswer:")
print(response["response"])