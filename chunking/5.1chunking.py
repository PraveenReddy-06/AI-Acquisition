text = """
FastAPI is a modern Python web framework.
It is used to build APIs quickly.
FastAPI supports automatic API documentation.
It uses Pydantic for data validation.
FastAPI can also work with asynchronous programming.
Python is widely used for artificial intelligence.
Machine learning uses data to learn patterns.
Deep learning uses neural networks.
"""

words = text.split()

chunk_size = 20

chunks = []

for i in range(0, len(words), chunk_size):
    chunk = words[i : i+chunk_size] #slicing words [0:19] i.e [i : i+chunk_size]
    chunks.append(" ".join(chunk))

for i in range(len(chunks)):
    print("Chunk", i + 1)
    print(chunks[i])
    print()
