from src.classifier import Classifier
from src.retriever import Retriever
from src.config import MODEL_PATH

def test_classifier_loads_model():
    clf = Classifier(MODEL_PATH)

    assert hasattr(clf, "model_path")
    assert hasattr(clf, "vectorizer")
    assert hasattr(clf, "mapping")

def test_classify_handles_empty_string():
    clf = Classifier(MODEL_PATH)
    label = clf.classify("")
    assert isinstance(label, str)
    assert label in clf.mapping.values()


def test_retrieve_basic_output():
    """
    Ensure retrieve() returns a list of strings with correct top_k length.
    """
    clf = Retriever()

    query = "I feel sad"
    chunks = clf.retrieve(text=query, top_k=3)

    assert isinstance(chunks, list)
    assert len(chunks) == 3
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c.strip()) > 0 for c in chunks)


def test_retrieve_semantic_relevance():
    """
    If the query expresses 'sadness', the top result should contain 
    sadness-related wording more often than unrelated emotions.
    This test does NOT assume an exact output — only that retrieval works logically.
    """
    clf = Retriever()

    query = "I feel very lonely and depressed today."
    chunks = clf.retrieve(text=query, top_k=1)
    top_chunk = chunks[0].lower()

    # soft semantic-meaning checks
    sadness_keywords = ["sad", "lonely", "depress", "grief"]
    assert any(word in top_chunk for word in sadness_keywords)


