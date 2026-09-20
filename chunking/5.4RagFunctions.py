import ollama
import math


def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


def create_embedding(text):
    response = ollama.embed(
        model="nomic-embed-text",
        input=text
    )
    return response["embeddings"][0]


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


def retrieve(question, chunks, k):
    question_embedding = create_embedding(question)
    results = []
    for chunk in chunks:
        chunk_embedding = create_embedding(chunk)
        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )
        results.append((chunk, score))
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )
    return results[:k]


def generate_answer(question, retrieved_chunks):
    context = ""
    for chunk, score in retrieved_chunks:
        context += chunk + "\n"

    prompt = f"""
    Answer the question using ONLY the context.

    If the answer is not present in the context,
    say "I don't know based on the provided context."

    Context:
    {context}

    Question:
    {question}
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
    return response["message"]["content"]


# -----------------------------
# MAIN
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

chunks = chunk_text(text, 20)
question = "What is used for data validation?"
retrieved_chunks = retrieve(
    question,
    chunks,
    2
)

print("\nRetrieved chunks:\n")

for chunk, score in retrieved_chunks:
    print("Score:", score)
    print("Chunk:", chunk)
    print()


answer = generate_answer(
    question,
    retrieved_chunks
)

print("Answer:")
print(answer)