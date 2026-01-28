import logging
import pickle
from src.config import MODEL_PATH
from src.label_mapping import load_label_mapping

class Classifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        logging.info(f"Loading classification pipeline from: {model_path}")
        with open(model_path, "rb") as f:
            self.clf_pipeline = pickle.load(f)
        logging.info("Classification pipeline loaded successfully.")

        self.vectorizer = self.clf_pipeline.named_steps['vectorizer']
        self.mapping = load_label_mapping()
        logging.info(f"Loaded label mapping: {self.mapping}")

    def classify(self, text: str) -> str:
        """Predicts the emotion of a given text."""
        # The pipeline handles vectorization and prediction in one step
        prediction = self.clf_pipeline.predict([text])
        label_id = int(prediction[0])
        emotion_name = self.mapping.get(label_id, "unknown") # Safe access
        return emotion_name
