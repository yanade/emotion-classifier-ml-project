from src.emotion_classifier import EmotionClassifier
from src.config import MODEL_PATH

def test_classifier_loads_model():
    clf = EmotionClassifier(MODEL_PATH)

    assert hasattr(clf, "model")
    assert hasattr(clf, "vectorizer")
    assert hasattr(clf, "mapping")