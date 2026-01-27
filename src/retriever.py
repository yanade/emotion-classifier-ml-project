import logging
import pickle
from src.config import SENTENCE_MODEL_PATH, SENTENCE_MODEL_NAME
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self):
        logging.info("Initializing knowledge retriever...")
        with open(SENTENCE_MODEL_PATH, "rb") as f:
            knowledge_data = pickle.load(f)
        self.knowledge_embeddings = knowledge_data["embeddings"]
        self.knowledge_chunks = knowledge_data["chunk"]
        logging.info("Knowledge embeddings and chunks loaded.")

        self.embedder = SentenceTransformer(SENTENCE_MODEL_NAME)
        logging.info("Sentence embedding model loaded.")

    def retrieve(self, text: str, top_k: int = 1) -> list:
        """Retrieves the most relevant knowledge chunks for a given text."""
        query_emb = self.embedder.encode([text])
        sims = cosine_similarity(query_emb, self.knowledge_embeddings)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        retrieved = [self.knowledge_chunks[i] for i in top_indices]
        return retrieved
