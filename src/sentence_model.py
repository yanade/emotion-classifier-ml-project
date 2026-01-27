from src.config import KNOWLEDGE_PATH, SENTENCE_MODEL_NAME, SENTENCE_MODEL_PATH 
import os
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import pickle
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)


embedding_model = SentenceTransformer(SENTENCE_MODEL_NAME)


def load_files(path=KNOWLEDGE_PATH):
    logging.info(f"Loading files from: {path}")
    documents = []

    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())
    logging.info(f"Loaded {len(documents)} documents")
    return documents

def chunk_knowledge(text):
    """Splits text into meaningful chunks based on sentences."""
    # Split by newlines and then further split by sentence-ending punctuation
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text.replace('\n', ' '))
    
    # Filter out any empty strings that may result from splitting
    chunks = [s.strip() for s in sentences if s.strip()]
    
    logging.info(f"Created {len(chunks)} chunks from text.")
    return chunks

def create_all_chunks(documents): 
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_knowledge(doc))
    logging.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks




def embed_knowledge(chunks_list):
    embeddings = embedding_model.encode(chunks_list)
    logging.info(f"Embedding matrix shape: {embeddings.shape}")
    return embeddings

if __name__ == "__main__":
    documents = load_files()
    all_chunks = create_all_chunks(documents)
    embeddings = embed_knowledge(all_chunks)



    
    with open(SENTENCE_MODEL_PATH , "wb") as f:
        pickle.dump(
            {"embeddings": embeddings,
            "chunk": all_chunks},f)
        logging.info(f"Model saved to {SENTENCE_MODEL_PATH }")