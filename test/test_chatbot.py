from src.simple_interface import clean_input
from src.language_model import Chatbot
from src.classifier import Classifier
from src.retriever import Retriever
from src.config import MODEL_PATH

def test_clean_text_removes_symbols():
    text = "Hello!!! ###World"
    cleaned = clean_input(text)
    assert "###" not in cleaned
    assert cleaned.strip() != ""
    assert cleaned == "hello!!! world"

def test_chatbot_basic_reply():
    classifier = Classifier(MODEL_PATH)
    retriever = Retriever
    bot = Chatbot(classifier=classifier, retriever=retriever)
    reply = bot.generate_reply("Hello!")
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0




def test_pipeline_full_flow():
    """
    End-to-end test:
    1. Classify user text
    2. Retrieve RAG chunks
    3. Pass both into chatbot
    """
    clf = Classifier(MODEL_PATH)
    retriever = Retriever()
    text = "I feel very sad today. Nothing is going right."
    label = clf.classify(text)   
    assert isinstance(label, str)
    assert label in clf.mapping.values()


    chunks = retriever.retrieve(text=text, top_k=2)
    assert isinstance(chunks, list)
    assert len(chunks) == 2

    bot = Chatbot(classifier=clf, retriever=retriever)
    bot.system_prompt = (
        f"Detected emotion: {label}. "
        f"Use this info in the response. "
        f"Helpful context: {' '.join(chunks)}"
    )
    reply = bot.generate_reply(text)
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0
    
