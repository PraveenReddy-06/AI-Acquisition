import ollama
import math


# -----------------------------
# 1. Document
# -----------------------------

text = """
FastAPI is a modern Python web framework.
It is used to build APIs quickly.
FastAPI supports automatic API documentation.
Praveen and venkey is used for data validation.
FastAPI supports asynchronous programming.
Python is widely used for artificial intelligence.
Machine learning uses data to learn patterns.
Deep learning uses neural networks.
"""


# -----------------------------
# 2. Chunking
# -----------------------------

words = text.split()

chunk_size = 20

chunks = []

for i in range(0, len(words), chunk_size):
    chunk = words[i:i + chunk_size]
    chunks.append(" ".join(chunk))


print("Chunks:")

for i in range(len(chunks)):
    print(i, ":", chunks[i])


# -----------------------------
# 3. Cosine similarity
# -----------------------------

def cosine_similarity(a, b):

    dot = 0
    mag_a = 0
    mag_b = 0

    for i in range(len(a)):
        dot += a[i] * b[i]
        mag_a += a[i] * a[i]
        mag_b += b[i] * b[i]

    mag_a = math.sqrt(mag_a)
    mag_b = math.sqrt(mag_b)

    return dot / (mag_a * mag_b)


# -----------------------------
# 4. Embed every chunk
# -----------------------------

chunk_embeddings = []

for chunk in chunks:

    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    chunk_embeddings.append(embedding)


# -----------------------------
# 5. Question
# -----------------------------

question = "What is used for data validation?"

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]


# -----------------------------
# 6. Calculate similarity
# -----------------------------

results = []

for i in range(len(chunk_embeddings)):

    score = cosine_similarity(
        question_embedding,
        chunk_embeddings[i]
    )

    results.append((chunks[i], score))


# -----------------------------
# 7. Rank results
# -----------------------------

results.sort(
    key=lambda x: x[1],
    reverse=True
)


# -----------------------------
# 8. Top-K
# -----------------------------

k = 2

print("\nTop", k, "relevant chunks:\n")

for i in range(k):

    print("Rank:", i + 1)
    print("Chunk:", results[i][0])
    print("Score:", results[i][1])
    print()


context=""
for i in range(k):
    context+=results[i][0]+"\n"


prompt=f"""
You are a question-answering system.
Answer the question using ONLY the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided context."
context:{context}
question:{question}
"""

response= ollama.chat(
    model="qwen2:0.5b",
    messages=[
        {
            "role":"user",
            "content":prompt
        }
    ]
)

print(response["message"]["content"])
