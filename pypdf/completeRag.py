import ollama
import math
from pypdf import PdfReader


# ==========================================
# 1. READ PDF
# ==========================================

pdf_path = "pypdf/notes.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print("Number of pages:", len(reader.pages))


# ==========================================
# 2. CHUNK PDF TEXT
# ==========================================

words = text.split()

chunk_size = 100
overlap = 20

chunks = []

start = 0

while start < len(words):

    end = start + chunk_size

    chunk = words[start:end]

    chunks.append(" ".join(chunk))

    start = end - overlap


print("Number of chunks:", len(chunks))


# ==========================================
# 3. COSINE SIMILARITY
# ==========================================

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


# ==========================================
# 4. CREATE EMBEDDINGS FOR CHUNKS
# ==========================================

chunk_embeddings = []

for chunk in chunks:

    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    chunk_embeddings.append(embedding)


print("Embeddings created:", len(chunk_embeddings))


# ==========================================
# 5. GET USER QUESTION
# ==========================================

question = "\n what is my project name and give its backend details"


# ==========================================
# 6. CREATE QUESTION EMBEDDING
# ==========================================

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)
print("Embedding response:", response)

if not response["embeddings"]:
    print("Failed to create question embedding.")
    exit()

question_embedding = response["embeddings"][0]


# ==========================================
# 7. CALCULATE SIMILARITY
# ==========================================

results = []

for i in range(len(chunk_embeddings)):

    score = cosine_similarity(
        question_embedding,
        chunk_embeddings[i]
    )

    results.append(
        (chunks[i], score)
    )


# ==========================================
# 8. SORT RESULTS
# ==========================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ==========================================
# 9. TOP-K
# ==========================================

k = 3

retrieved_chunks = results[:k]


print("\nRetrieved chunks:\n")

for i in range(len(retrieved_chunks)):

    print("Rank:", i + 1)
    print("Score:", retrieved_chunks[i][1])
    print("Chunk:")
    print(retrieved_chunks[i][0])
    print("--------------------------------")


# ==========================================
# 10. BUILD CONTEXT
# ==========================================

context = ""

for chunk, score in retrieved_chunks:

    context += chunk + "\n\n"


# ==========================================
# 11. CREATE RAG PROMPT
# ==========================================

prompt = f"""
You must answer the user's question using the CONTEXT below.

IMPORTANT RULES:
1. Ignore your previous knowledge.
2. Do not talk about yourself.
3. Do not say you are an AI assistant.
4. Use only information found in CONTEXT.
5. If the answer cannot be found in CONTEXT, say:
"I don't know based on the provided document."
6. Give a concise answer.

Context:
{context}

Question:
{question}

Answer:
"""


# ==========================================
# 12. SEND TO OLLAMA
# ==========================================

response = ollama.chat(
    model="qwen2:0.5b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# 13. FINAL ANSWER
# ==========================================

answer = response["message"]["content"]

print("\n================================")
print("ANSWER")
print("================================")
print(answer)