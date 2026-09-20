import ollama
import math

text1="i love nlp"
text2="i love nlk"

response1 = ollama.embed(model="nomic-embed-text",input=text1)
a=response1["embeddings"][0]

response2=ollama.embed(model="nomic-embed-text",input=text2)
b=response2['embeddings'][0]

ans=0
for j in range(len(a)):
    ans += a[j]*b[j]
print("Dot:" ,ans)

sqa=0
for i in a:
    sqa += i*i
sqa=math.sqrt(sqa)
print("mag a:", sqa)

sqb=0
for i in b:
    sqb +=i*i
sqb=math.sqrt(sqb)
print("mag b:", sqb)

print(ans/(sqa*sqb))






