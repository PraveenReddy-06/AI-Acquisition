import ollama
from pydantic import BaseModel


# Step 1: Define the expected output structure
class InternshipInfo(BaseModel):
    duration: str
    domain: str


# Step 2: Provide information to extract
text = """
The program was an 8-week Python Full Stack Development internship.
"""


# Step 3: Ask Ollama to return structured data
response = ollama.chat(
    model="qwen2:0.5b",
    messages=[
        {
            "role": "system",
            "content": (
                "Extract information from the given text. "
                "Return 'unknown' if information is missing."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ],
    format=InternshipInfo.model_json_schema()
)


# Step 4: Parse and validate the response
result = InternshipInfo.model_validate_json(
    response["message"]["content"]
)


# Step 5: Access individual values
print("Duration:", result.duration)
print("Domain:", result.domain)