from fastapi import FastAPI
import ollama
from pydantic import BaseModel

app=FastAPI()

class chatbot(BaseModel):
    userId:int
    question :str


notes = [
    "Merge Sort is a divide and conquer algorithm.",
    "Quick Sort uses a pivot element.",
    "12Heap Sort uses a binary heap."
]

@app.post("/ask")
def chat(request: chatbot):

    context=""

    for note in notes:
        if request.question.lower() in note.lower():
            context=note
            break
        
    prompt = f"""
    Context:{context}
    Question:{request.question}
    """

    response = ollama.chat(
        model="qwen2:0.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "answer":response["message"]["content"]
    }





