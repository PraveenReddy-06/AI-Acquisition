import ollama

response = ollama.chat(
    model="qwen2:0.5b",
    messages=[
        {
            "role":"user",
            "content":"what is jwt"
        }
    ]
)

print(response["message"]["content"])