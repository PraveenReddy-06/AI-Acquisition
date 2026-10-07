#The problem is that vector databases are optimized for fast retrieval, not perfect relevance judgment.
# so we just rerank them 

# Why not use the reranker directly?
# Because reranking is generally more expensive than vector search.


#If we have 100,000 chunks:
# Vector search
# 100,000 → 20
# is very fast.

# Then:
# Reranker
# 20 → 5
# is manageable.

# Doing:
# 100,000 → reranker
# would be unnecessarily expensive.

import chromadb
import ollama

client = chromadb.PersistentClient(path="./chroma_data")
collections = client.list_collections()
for col in collections:
    print(col.name)

collection = client.get_collection(name="pdf_documents")


question = input("\nAsk a question: ")

response = ollama.embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=10
)

question_words = question.lower().split()
hybrid_results = []


for i in range(len(results["documents"][0])):
    document = results["documents"][0][i]
    distance = results["distances"][0][i]
    metadata = results["metadatas"][0][i]
    semantic_score = distance
    document_words = document.lower().split()
    keyword_matches = 0
    for word in question_words:
        if word in document_words:
            keyword_matches += 1
    if len(question_words) > 0:
        keyword_score = (keyword_matches / len(question_words))
    else:
        keyword_score = 0

    final_score = (0.7 * semantic_score + 0.3 * keyword_score)

    hybrid_results.append({
        "document": document,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "final_score": final_score,
        "metadata": metadata
    })

hybrid_results.sort(
    key=lambda x: x["final_score"],
    reverse=True
)


print("\nHybrid Search Results:\n")
top_k = 3
for i, result in enumerate(hybrid_results[:top_k],start=1):

    print("Rank:", i)
    print(
        "Semantic Score:",
        round(result["semantic_score"])
    )
    print(
        "Keyword Score:",
        round(result["keyword_score"], 4)
    )
    print(
        "Final Score:",
        round(result["final_score"], 4)
    )
    print(
        "Source:",
        result["metadata"]
    )
    print(
        "Page:",
        result["metadata"]["page"]
    )
    print(
        "Chunk:",
        result["metadata"]["chunk"]
    )
    print(
        "Text:",
        result["document"]
    )

    print("-" * 70)