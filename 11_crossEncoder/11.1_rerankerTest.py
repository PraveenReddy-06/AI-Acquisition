from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

question = "who is brave?"

documents = [
    "Praveen offers an 8-week Python Full Stack Development internship with hands-on project work.",
    "The internship certificate was issued on August 3, 2026.",
    "Jas is bold",
    "Jaanu is my cute potttaaatoeee."
]

pairs = []

for document in documents:
    pairs.append(
        [question, document]
    )

scores = model.predict(pairs)

for i in range(len(documents)):
    print( "Score:", scores[i])
    print( "Document:",documents[i] )
    print("-" * 60)