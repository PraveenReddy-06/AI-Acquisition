import ollama
from fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class chatRequest(BaseModel):
    question:str

messages=[]

@app.post("/ask")
def chat(request:chatRequest):

    messages.append({
        "role": "user",
        "content":request.question
    })

    response= ollama.chat(
        model="qwen2:0.5b",
        messages= messages
    )

    messages.append({
        "role":"assistant",
        "content":response["message"]["content"]
    })
    print(messages)
    return {
        "answer": response["message"]["content"]
    }
