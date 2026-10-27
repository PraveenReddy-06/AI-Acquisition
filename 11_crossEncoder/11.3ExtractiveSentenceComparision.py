                #     USER QUESTION
                #           │
                #           ▼
                #  nomic-embed-text
                #           │
                #           ▼
                #       ChromaDB
                #           │
                #      Top 20
                #           │
                #           ▼
                #   Cross-Encoder
                #     Reranking
                #           │
                #       Top 3
                #           │
                #           ▼
                # Context Compression
                #           │
                #           ▼
                #      RAG Prompt
                #           │
                #           ▼
                #        Qwen2
                #           │
                #           ▼
                #        Answer
#words don't necessarily represent semantic relevance.


def compress_context(question, document):
    
    question_words = set(
        question.lower().split()
    )

    sentences = document.split(".")
    relevant_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        #is a set intersection.
        common_words = (question_words & sentence_words)
        score = len(common_words)
        if score > 0:
            relevant_sentences.append((sentence, score))

    relevant_sentences.sort(
        key=lambda x: x[1],
        reverse=True
    )

    compressed = []
    for sentence, score in relevant_sentences:
        compressed.append(sentence.strip())

    return ". ".join(compressed)