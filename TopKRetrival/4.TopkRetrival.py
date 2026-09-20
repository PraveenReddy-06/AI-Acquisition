import ollama
import math

documents = [
    "he eats gulamjamun",
    "ram like mitais",
    "he love blue "
]

question = "what colour does praveen like?"
#here we will get "he  love blue" as answer even there is like in question which should give "ram like mitais"

# 1. Cosine similarity function
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


# 2. Create question embedding
response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]


# 3. Create document embeddings
document_embeddings = []

for document in documents:

    response = ollama.embed(
        model="nomic-embed-text",
        input=document
    )

    embedding = response["embeddings"][0]

    document_embeddings.append(embedding)


# 4. Calculate similarity for every document
results = []

for index in range(len(document_embeddings)):

    score = cosine_similarity(
        question_embedding,
        document_embeddings[index]
    )

    results.append((documents[index], score))


# 5. Sort by similarity score
results.sort(key=lambda x: x[1], reverse=True)


# 6. Display Top-K
k = 2

print("\nTop", k, "results:\n")

for i in range(k):
    document = results[i][0]
    score = results[i][1]

    print("Rank:", i + 1)
    print("Document:", document)
    print("Score:", score)
    print()