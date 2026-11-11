import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Setup
print("Loading embedding model and database...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
index = faiss.read_index("tax_index.faiss")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Groq Configuration
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), # This is your Groq key (gsk_...)
    base_url="https://api.groq.com/openai/v1" 
)

# REFUSAL GUARDRAIL: If distance is too high (similarity too low), refuse.
# FAISS L2 distance: Lower is better. Tune this based on your tests.
SIMILARITY_THRESHOLD = 1.1 

def answer_question(question):
    # 1. Embed question
    q_emb = model.encode([question]).astype('float32')
    
    # 2. Search FAISS
    distances, indices = index.search(q_emb, 4) # Get top 4 chunks
    
    best_distance = distances[0][0]
    retrieved_chunks = [metadata[i] for i in indices[0]]
    
    # 3. PRE-REFUSAL GUARDRAIL
    if best_distance > SIMILARITY_THRESHOLD:
        return {
            "answer": "I don't have enough information to answer this based on the provided provisions.",
            "refused": True,
            "reason": f"Out of scope (distance {best_distance:.2f})",
            "chunks_used": []
        }

    # 4. Build Context
    context = "\n\n".join([f"[{c['section']}] {c['text']}" for c in retrieved_chunks])
    
    # 5. Prompt the LLM
    prompt = f"""You are an Indian income tax Q&A assistant.
Answer the question using ONLY the context below.
If the context does not contain enough information to answer, respond exactly:
"I don't have enough information to answer this based on the provided provisions."
Do not use your own knowledge. Do not guess.

CRITICAL: If the question asks if you can claim multiple things (e.g., both 80C and 80D), and the context contains information on BOTH sections, you ARE allowed to say "Yes" and summarize the rules of both. Synthesize the information provided.

When you answer, cite the section number from the context.

Context:
{context}

Question: {question}
"""
    
    # 6. Call Groq API
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b", # Updated to Groq's current model
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    
    answer_text = resp.choices[0].message.content.strip()
    
    # 7. POST-REFUSAL CHECK (Did the LLM refuse even if retrieval passed?)
    refused_by_llm = "don't have enough information" in answer_text.lower()
    
    return {
        "answer": answer_text,
        "refused": refused_by_llm,
        "reason": "LLM refused based on context" if refused_by_llm else "Success",
        "chunks_used": [c['section'] for c in retrieved_chunks]
    }