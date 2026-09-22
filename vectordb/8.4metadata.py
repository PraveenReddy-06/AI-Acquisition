import chromadb
import ollama
from pypdf import PdfReader

reader = PdfReader("pypdf/notes.pdf")

text=""
for page in reader.pages:
    if page.extract_text():
        text+=page.extract_text()

# page.extract_text()	String	"Car runs on road"
# text += page.extract_text()	String	"Car runs on road Plane flies in sky................"
# text.split()	List of strings	["Car", "runs", "on", "road"]
# words[start:start+chunk_size]	List of strings	["Car", "runs", "on", "road"]
# " ".join(words[start:start+chunk_size])	String	"Car runs on road"

words=text.split()
chunks=[]
chunk_size=5
overlap=2
start=0
while start<len(words):
    chunk=" ".join(words[start:start+chunk_size])
    chunks.append(chunk)
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



question="how man weeks is this internship"
question_embed=ollama.embed(
    model="nomic-embed-text",
    input=question
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

# PDF
#  ↓
# Extract text
#  ↓
# Split into chunks
#  ↓
# nomic-embed-text
#  ↓
# Store embeddings in ChromaDB
#  ↓
# Question
#  ↓
# Question embedding
#  ↓
# ChromaDB query
#  ↓
# Top 2 chunks
