from pypdf import PdfReader

reader = PdfReader("pypdf/notes.pdf")
text=""

for page in reader.pages:
    page_text=page.extract_text()
# Why do we need this?
# Some PDF pages may have no extractable text.
# For example:
# page_text = None
# If we blindly do:
# text += page_text
# Python can produce an error because None isn't a string.
    if page_text:
        text+=page_text+"\n"

print("no of pages:", len(reader.pages))
print("text:\n" ,text)
