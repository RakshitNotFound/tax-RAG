import os
import faiss
import pickle
import glob
from sentence_transformers import SentenceTransformer

# Use a fast, free local embedding model
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chunks = []
metadata = []

# Read all text files in the corpus folder
for filepath in glob.glob("corpus/*.txt"):
    filename = os.path.basename(filepath)
    section_name = filename.replace(".txt", "")
    
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Simple chunking: split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    
    for p in paragraphs:
        chunks.append(p)
        metadata.append({"section": section_name, "text": p})

print(f"Created {len(chunks)} chunks from {len(glob.glob('corpus/*.txt'))} files.")

# Create embeddings
print("Embedding chunks...")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = embeddings.astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index, metadata, and model name
faiss.write_index(index, "tax_index.faiss")
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Index built and saved successfully!")