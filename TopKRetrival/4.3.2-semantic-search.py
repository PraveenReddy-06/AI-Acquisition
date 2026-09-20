import ollama
import math

documents = [
    "praveen like gulamjamun",
    "ram like mitais",
    "praveen like blue colour"
]

question="what colour does praveen like?"

response_embedding=ollama.embed(model="nomic-embed-text",input=question)
question_embedding=response_embedding["embeddings"][0]
question_mag=0
for i in range(len(question_embedding)):
    question_mag += question_embedding[i]*question_embedding[i]
question_mag=math.sqrt(question_mag)


document_embeddings = []
for document in documents:
    response = ollama.embed(
        model="nomic-embed-text",
        input=document
    )
    embedding = response["embeddings"][0]
    document_embeddings.append(embedding)


final_ans=""
maxscore=0
for document in document_embeddings:
    score=0
    dot=0
    mag=0
    for i in range(len(document)):
        dot+=document[i]*question_embedding[i]
        mag+=document[i]*document[i]
    mag=math.sqrt(mag)
    score=dot/(mag*question_mag)
    if(score>maxscore):
        maxscore=score
        final_ans=document

print("max score",maxscore)
print(final_ans)

