# aiStudymate_cpu.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------
# Setup FastAPI
# -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------
# Load SentenceTransformer + FAISS
# -------------------
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

index_file = "faiss_index_cpu.index"
meta_file = "meta_cpu.pkl"

if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM
    metadata = []

# -------------------
# Load small CPU-friendly LLM
# -------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")

# -------------------
# Pydantic model
# -------------------
class Query(BaseModel):
    question: str
    k: int = 3   # default = 3

# -------------------
# API endpoints
# -------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    doc = fitz.open(path)
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            vector = embedding_model.encode([text])
            index.add(np.array(vector).astype('float32'))
            metadata.append({"text": text, "filename": file.filename})

    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(metadata, f)

    return {"status": "success", "filename": file.filename}


@app.post("/ask")
async def ask_question(query: Query):
    question_vec = embedding_model.encode([query.question])
    D, I = index.search(np.array(question_vec).astype('float32'), k=query.k)

    # ✅ Fix: prevent -1 index issue
    retrieved_texts = [metadata[i]["text"] for i in I[0] if 0 <= i < len(metadata)]

    # ✅ Better prompt
    context = "\n---\n".join(retrieved_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {query.question}\nAnswer in a clear and concise way:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=150, do_sample=True, top_p=0.9)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"answer": answer}
