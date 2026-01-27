import logging
import pickle
from src.config import MODEL_PATH, SENTENCE_MODEL_PATH, SENTENCE_MODEL_NAME
from src.label_mapping import load_label_mapping
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

class EmotionClassifier:
    def __init__(self, model_path=MODEL_PATH):
        logging.info(f"Loading model from: {MODEL_PATH}")

        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        self.vectorizer = saved["vectorizer"]
        self.model = saved["model"]
        logging.info("Model and vectorizer loaded successfully.")

        self.mapping = load_label_mapping()
        logging.info(f"Loaded label mapping: {self.mapping}")

        with open(SENTENCE_MODEL_PATH, "rb") as f:
            knowledge_data = pickle.load(f)
        self.knowledge_embeddings = knowledge_data["embeddings"]
        self.knowledge_chunks = knowledge_data["chunk"]
        logging.info("Knowledge embeddings and chunks loaded.")
        self.embedder = SentenceTransformer(SENTENCE_MODEL_NAME)
        logging.info("Sentence embedding model loaded.")
    
    
    
    def classify(self, text: str):
        vec = self.vectorizer.transform([text])
        prediction = self.model.predict(vec)
        label_id = int(prediction[0])
        emotion_name = self.mapping[label_id]
        return emotion_name
    
    
    #Retrieve the most relevant chunks
    def retrieve(self, text: str, top_k: int = 2):
        query_emb = self.embedder.encode([text])
        sims = cosine_similarity(query_emb, self.knowledge_embeddings)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        retrieved = [self.knowledge_chunks[i] for i in top_indices]
        return retrieved
