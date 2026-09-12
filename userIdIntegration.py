import ollama
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class chatbot(BaseModel):
    userId:int
    question:str

chat_memory={}

@app.post("/ask")
def ask(chatbot: chatbot):

    if chatbot.userId not in chat_memory:
        chat_memory[chatbot.userId]=[]

    chat_memory[chatbot.userId].append({
        "role":"user",
        "content":chatbot.question
    })

    response=ollama.chat(
        model="qwen2:0.5b",
        messages=chat_memory[chatbot.userId]
    )

    chat_memory[chatbot.userId].append({
        "role":"assistant",
        "content":response["message"]["content"]
    })

    return {
        "answer":response["message"]["content"]
    }


