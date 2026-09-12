from fastapi import FastAPI

app=FastAPI()

@app.get("/student/{id}")
def hello(id : int):
    return {
       "id":id
    }
