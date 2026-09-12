from fastapi import FastAPI
import ollama
from pydantic import BaseModel

app= FastAPI()

class chatbot(BaseModel):
    userid:int
    question:str

notes = [
    "Merge Sort is a divide and conquer algorithm.",
    "Quick Sort uses a pivot element.",
    "Heap Sort uses a binary heap."
]
context=""

for note in notes:
    if "quick sort" in note.lower():
        context=note
        break

print(context)
