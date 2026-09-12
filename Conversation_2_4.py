import ollama
from pydantic import BaseModel
from fastapi import FastAPI

app=FastAPI()

class chatbot(BaseModel):
    userId:int
    question:str

corpus=[
    "Praveen is a badboy but he is good coder",
    "gireesh is very smart boy",
    "jasmine love flowers and she is very pretty."
]


def retrive(question):
    context=""
    maxscore=0    
    questionwords = question.lower()
    for c in corpus:
        score=0
        for q in questionwords.split():
            if q in c:
                score=score+1
        if score>maxscore:
            maxscore=score
            context=c
    return context

memory={}

@app.put("/ask")
def ask(chatbot:chatbot):
    if chatbot.userId not in memory:
        memory[chatbot.userId]=[]
    
    context = retrive(chatbot.question)

    prompt = f""" You are a RAG question-answering system.You MUST answer using ONLY the information in the Context.Rules:1. Do not use your own knowledge.2. Do not invent facts.3. Do not add information that is not in the Context.4. If the answer is not available in the Context, say:"I don't know based on the given data."
    Retrived Context:{context}
    current Question:{chatbot.question}
    """

    message=memory[chatbot.userId].copy()
    message.append({
        "role":"user",
        "content":prompt
    })

    response=ollama.chat(
        model="qwen2:0.5b",
        messages=message
    )

    answer= response["message"]["content"]

    memory[chatbot.userId].append({"role":"user","content":chatbot.question})
    memory[chatbot.userId].append({"role":"assistant","content":answer})

    return {
        "userId": chatbot.userId,
        "question": chatbot.question,
        "context": context,
        "answer": answer
    }