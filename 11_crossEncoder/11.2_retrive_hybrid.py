from sentence_transformers import CrossEncoder
import chromadb
import ollama


# -----------------------------
# 1. Load ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_collection(
    name="pdf_documents"
)


# -----------------------------
# 2. Load reranker
# -----------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


# -----------------------------
# 3. Question
# -----------------------------

question = input("\nAsk a question: ")


# -----------------------------
# 4. Question embedding
# -----------------------------

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]


# -----------------------------
# 5. Retrieve candidates
# -----------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=20
)


documents = results["documents"][0]
metadatas = results["metadatas"][0]


# -----------------------------
# 6. Create question-document
#    pairs
# -----------------------------

pairs = []

for document in documents:

    pairs.append([
        question,
        document
    ])


# -----------------------------
# 7. Rerank
# -----------------------------

scores = reranker.predict(pairs)


# -----------------------------
# 8. Combine documents + scores
# -----------------------------

reranked_results = []

for i in range(len(documents)):

    reranked_results.append({

        "document": documents[i],

        "score": float(scores[i]),

        "metadata": metadatas[i]

    })


# -----------------------------
# 9. Sort by reranker score
# -----------------------------

reranked_results.sort(
    key=lambda x: x["score"],
    reverse=True
)


# -----------------------------
# 10. Display Top 3
# -----------------------------

print("\nReranked Results:\n")


for i, result in enumerate(reranked_results[:3],start=1):

    print("Rank:", i)
    print(
        "Reranker Score:",
        round(result["score"], 4)
    )
    print(
        "Source:",
        result["metadata"].get("source")
    )
    print(
        "Page:",
        result["metadata"].get("page")
    )
    print(
        "Chunk:",
        result["metadata"].get("chunk")
    )
    print(
        "Text:",
        result["document"]
    )
    print("-" * 70)


context = ""
for result in reranked_results[:3]:
    context += (
        f"Source: {result['metadata'].get('source')}\n"
        f"Page: {result['metadata'].get('page')}\n"
        f"Content: {result['document']}\n\n"
    )

prompt = f"""
Answer the question using only the context below.
CONTEXT:
{context}
QUESTION:
{question}
ANSWER:
"""

response = ollama.chat(
    model="qwen2:0.5b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
print("answer from ohohohllllama:\n",response["message"]["content"])