from pypdf import PdfReader


pdf_path = "notes.pdf"

reader = PdfReader(pdf_path)

chunks = []
chunk_size = 100
overlap = 20
chunk_id = 0
for page_number, page in enumerate(reader.pages):
    page_text = page.extract_text()
    if not page_text:
        continue
    words = page_text.split()
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": chunk_id,
            "page": page_number + 1,
            "text": chunk_text
        })
        chunk_id += 1
        start = end - overlap
print("Total chunks:", len(chunks))


for chunk in chunks[:3]:

    print("\nChunk ID:", chunk["id"])
    print("Page:", chunk["page"])
    print("Text:", chunk["text"])