# Tax-Provision RAG with Refusal Guardrails

An AI-powered Q&A assistant built for Indian Income Tax provisions. 

In tax software, a confidently wrong answer is worse than useless. This project focuses heavily on **guardrails** to prevent hallucinated deductions and **evaluation** to measure exactly when the system refuses to answer.

## 🛠️ Tech Stack
*   **LLM:** Groq API (`gpt-oss-20b` model)
*   **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5` (Local, CPU-friendly)
*   **Vector Database:** FAISS (Facebook AI Similarity Search)
*   **Backend:** Python

## 🏗️ Architecture & Guardrails
1.  **Ingestion:** Legal text from the Income Tax Act (Sections 80C, 80D, HRA, Standard Deduction) is chunked and embedded into a local FAISS index.
2.  **Retrieval:** User queries are embedded and searched against the FAISS index to find the top 4 most relevant legal chunks.
3.  **Pre-LLM Refusal Guardrail:** If the vector distance (similarity score) of the retrieved chunks is below a specific threshold, the system immediately refuses to answer before even calling the LLM. This saves API costs and blocks out-of-scope questions.
4.  **Prompt Guardrail:** The LLM is strictly instructed to only use the provided context and to output a specific refusal phrase if the answer isn't present.
5.  **Post-LLM Evaluation:** A custom test harness runs 3 buckets of questions to grade the system's behavior.

## 📊 Evaluation Results
The system was evaluated on 6 test cases across three distinct buckets:

*   **Answerable (100% Pass):** e.g., "What is the 80C limit?" -> Correctly answers ₹150,000.
*   **Not in Corpus (100% Pass):** e.g., "Can I deduct crypto losses?" -> Correctly refuses to answer, preventing hallucinations.
*   **Ambiguous/Multi-Hop (50% Pass):** e.g., "How is HRA calculated alongside standard deduction?" -> The FAISS search retrieved the standard deduction but missed the HRA chunk, prompting a safe refusal. 

## 🚀 How to Run Locally

**1. Install dependencies:**
```bash
pip install sentence-transformers faiss-cpu openai python-dotenv
