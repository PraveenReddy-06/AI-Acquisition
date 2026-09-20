import ollama
import math

documents = [
    "praveen like gulamjamun",
    "ram like mitais",
    "praveen like blue colour"
]

question="what colour does praveen like?"

response_embedding=ollama.embed(model="nomic-embed-text",input=question)
embedding=response_embedding["embeddings"][0]

response_doc1=ollama.embed(model="nomic-embed-text",input=documents[0])
embedding_doc1=response_doc1["embeddings"][0]
response_doc2=ollama.embed(model="nomic-embed-text",input=documents[1])
embedding_doc2=response_doc2["embeddings"][0]
response_doc3=ollama.embed(model="nomic-embed-text",input=documents[2])
embedding_doc3=response_doc3["embeddings"][0]

question_mag=0
for i in range(len(embedding)):
    question_mag += embedding[i]*embedding[i]
question_mag=math.sqrt(question_mag)



doc1dot=0
for j in range(len(embedding_doc1)):
    doc1dot += embedding_doc1[j]*embedding[j]
doc1_mag=0
for i in range(len(embedding_doc1)):
    doc1_mag +=embedding_doc1[i]*embedding_doc1[i]
doc1_mag=math.sqrt(doc1_mag)
print("score for dco1:",doc1dot/(question_mag*doc1_mag))



doc2dot=0
for j in range(len(embedding_doc2)):
    doc2dot += embedding_doc2[j]*embedding[j]
doc2_mag=0
for i in range(len(embedding_doc2)):
    doc2_mag +=embedding_doc2[i]*embedding_doc2[i]
doc2_mag=math.sqrt(doc2_mag)
print("score for dco2:",doc2dot/(question_mag*doc2_mag))



doc3dot=0
for j in range(len(embedding_doc3)):
    doc3dot += embedding_doc3[j]*embedding[j]
doc3_mag=0
for i in range(len(embedding_doc3)):
    doc3_mag +=embedding_doc3[i]*embedding_doc3[i]
doc3_mag=math.sqrt(doc3_mag)
print("score for dco3:",doc3dot/(question_mag*doc3_mag))

