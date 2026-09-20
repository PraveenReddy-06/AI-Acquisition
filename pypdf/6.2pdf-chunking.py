from pypdf import PdfReader

reader= PdfReader("pypdf/notes.pdf")

text=""
for page in reader.pages:
    if page.extract_text():
        text+= page.extract_text() + "\n"

words=text.split()
chunksize=20
overlap=3
start=0
tokens=[]
while start<len(words):
    token=words[start:start+chunksize]
    tokens.append(token)
    start=start+chunksize-overlap

print("Number of pages:", len(reader.pages))
print("Number of words:", len(words))
print("Number of chunks:", len(tokens))

print("\nFirst chunk:\n")
print(tokens[0])
    