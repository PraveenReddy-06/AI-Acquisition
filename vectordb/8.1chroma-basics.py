import chromadb

client = chromadb.Client()

collection = client.create_collection(name="query")

documents=[ "Car runs on land", "Plane flies in the sky", "Boat travels on water", "Bus is public transport on road"]
ids=["car1", "plane1", "boat1", "bus1"]

collection.add(documents=documents,ids=ids)

results=collection.query(
    query_texts=["which vehicle can fit 200 people"],
    n_results=2
)

print(results)