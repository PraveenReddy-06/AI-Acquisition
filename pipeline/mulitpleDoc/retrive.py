import chromadb
import ollama


# --------------------------------
# 1. Connect to existing database
# --------------------------------

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_collection(
    name="pdf_documents"
)


# --------------------------------
# 2. Ask question
# --------------------------------

question = input("\nAsk a question: ")


# --------------------------------
# 3. Optional document filter
# --------------------------------

source = input(
    "Enter document name or press Enter for all documents: "
)


# --------------------------------
# 4. Create question embedding
# --------------------------------

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]


# --------------------------------
# 5. Build query
# --------------------------------

query_parameters = {
    "query_embeddings": [question_embedding],
    "n_results": 3
}


# Add filter only if user selected a document
if source:

    query_parameters["where"] = {
        "source": source
    }


# --------------------------------
# 6. Search ChromaDB
# --------------------------------

results = collection.query(
    **query_parameters
)


# --------------------------------
# 7. Display results
# --------------------------------

print("\nRetrieved documents:\n")

for i in range(len(results["documents"][0])):

    print("Rank:", i + 1)

    print(
        "Distance:",
        results["distances"][0][i]
    )

    print(
        "Source:",
        results["metadatas"][0][i]["source"]
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