from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()

# ✅ Allow all origins (or restrict to http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👇 Your original code...
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "The Eiffel Tower is located in Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a programming language used for data science.",
    "The capital of India is New Delhi.",
    "Transformers are deep learning models for sequence tasks."
]
embeddings = model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

class Query(BaseModel):
    question: str

@app.post("/search")
def search(query: Query):
    query_vector = model.encode([query.question])
    distances, indices = index.search(query_vector, k=3)
    results = [documents[i] for i in indices[0]]
    return {"results": results}
