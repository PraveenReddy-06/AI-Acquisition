text = """
FastAPI is a modern Python web framework.
It is used to build APIs quickly.
FastAPI supports automatic API documentation.
Pydantic is used for data validation.
FastAPI supports asynchronous programming.
Python is widely used for artificial intelligence.
Machine learning uses data to learn patterns.
Deep learning uses neural networks.
"""

words = text.split()
chunk_size = 10
overlap = 3

chunks = []
start = 0

while start < len(words):
    end = start + chunk_size
    chunk = words[start:end]
    chunks.append(" ".join(chunk))
#this is important
    start = end - overlap


for i in range(len(chunks)):
    print("Chunk", i + 1)
    print(chunks[i])
    print()