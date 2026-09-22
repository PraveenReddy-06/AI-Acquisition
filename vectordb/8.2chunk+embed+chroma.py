import chromadb
import ollama


documents= [
  "Car runs on land",
  "Plane flies in the sky",
  "Boat travels on water",
  "Bus is public transport on road",
  "Train runs on railway tracks",
  "Helicopter hovers in the air",
  "Truck carries goods on highways",
  "Cycle is pedaled on roads",
  "Submarine moves under the sea",
  "Metro travels underground in tunnels",
  "Rocket launches into space",
  "Taxi carries passengers in the city",
  "Horse gallops on land",
  "Tram runs on tracks in town"
]

text=[]
for doc in documents:
    text.append(doc)

chunks=[]
chunk_size=5
overlap=2
start=0
while start<len(text):
    chunk=text[start:start+chunk_size]
    chunks+=chunk
    start=start+chunk_size-overlap

ids=[]
for i in range(len(chunks)):
    ids.append(f"chunk_{i}")

embeddings=[]
for chunk in chunks:
    embedding= ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )
    embeddings.append(embedding["embeddings"][0])



question="which vehicle runs on road"
question_embed=ollama.embed(
    model="nomic-embed-text",
    input=text
)
question_embed=question_embed["embeddings"][0]



client = chromadb.Client()
collection=client.create_collection(name="mycollection")
collection.add(documents=chunks,ids=ids,embeddings=embeddings)
results=collection.query(
    query_embeddings=question_embed,
    n_results=2
)

print(results)

