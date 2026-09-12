from fastapi import FastAPI
import ollama

app = FastAPI()

@app.get("/ask/{question}")
def ask(question: str):
    response = ollama.chat(
        model="qwen2:0.5b",
        messages=[{
            "role":"user",
            "content": question
        }
        ]
    )
    return{
        "answer":response["message"]["content"]
    }


