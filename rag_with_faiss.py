# Importing
import os
import numpy as np
import faiss

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# --------------------------------------------------
# 3. Load document
# --------------------------------------------------

file_path = r"C:\Users\divya\OneDrive\Documents\gen_ai_notes.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


# --------------------------------------------------
# 4. Create chunks
# --------------------------------------------------

chunk_size = 50

chunks = []

j = 1

for i in range(0, len(text[:1500]), chunk_size):

    chunk = {}

    chunk["text"] = text[i:i + chunk_size]

    chunk["metadata"] = {
        "source": "gen_ai_notes.txt",
        "page": j,
        "document_id": "doc_01",
        "title": "GenAI Notes"
    }

    j += 1

    chunks.append(chunk)


# --------------------------------------------------
# 5. Extract only text from chunks
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]


# --------------------------------------------------
# 6. Create embedding function
# --------------------------------------------------

def embed_texts(texts):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    embeddings = np.array(
        [item.values for item in response.embeddings],
        dtype="float32"
    )

    return embeddings


# --------------------------------------------------
# 7. Create embeddings for document chunks
# --------------------------------------------------

embeddings = embed_texts(texts)


# --------------------------------------------------
# 8. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# --------------------------------------------------
# 9. Take user question
# --------------------------------------------------

user_prompt = input("Enter your prompt: ")


# --------------------------------------------------
# 10. Create embedding for user question
# --------------------------------------------------

query_vector = embed_texts([user_prompt])


# --------------------------------------------------
# 11. Search FAISS
# --------------------------------------------------

k = 2

distances, indices = index.search(
    query_vector,
    k
)


# --------------------------------------------------
# 12. Retrieve matching chunks
# --------------------------------------------------

retrieved_chunks = [
    chunks[i]
    for i in indices[0]
]


# --------------------------------------------------
# 13. Create context
# --------------------------------------------------

context = "\n\n".join(
    chunk["text"]
    for chunk in retrieved_chunks
)


# --------------------------------------------------
# 14. Create RAG prompt
# --------------------------------------------------

prompt = f"""
You are a helpful assistant.

Answer the user's question using the provided context.

--- CONTEXT ---
{context}
--- END CONTEXT ---

--- QUESTION ---
{user_prompt}
--- END QUESTION ---

Instructions:
- Use the provided context.
- Do not invent information.
- If the answer is not present in the context,
  say that you don't have enough information.
"""


# --------------------------------------------------
# 15. Send context + question to Gemini
# --------------------------------------------------

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)


# --------------------------------------------------
# 16. Print answer
# --------------------------------------------------

print("\nAnswer:")
print(response.text)
