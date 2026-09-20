import ollama

text = "FastAPI is a Python framework for building APIs."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print(len(embedding))
print(embedding[10:])

