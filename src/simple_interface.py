import re
from src.emotion_classifier import EmotionClassifier
from src.config import MODEL_PATH

def clean_text(raw_text: str) -> str:
    raw_text = raw_text.lower()
    raw_text = raw_text.strip()
    raw_text = re.sub(r"[^\w\s.,!?']", " ", raw_text)
    raw_text = re.sub(r"#(\w+)", r"\1", raw_text)  
    raw_text = re.sub(r"\s+", " ", raw_text)
    return raw_text

def main():
    print(" Hello, I am Emotion classifier - simple interface.")
    print("Type :q or exit to leave.\n")
    print("Type a sentence to classify.\n")

    classifier = EmotionClassifier(MODEL_PATH)

    exit_conditions = ("q", "exit")
    while True:
        user_text = input("Enter a sentence: ").strip()
        
        if user_text in exit_conditions:
            print("Goodbye!")
            break

        if not user_text:
            print("Please enter non-empty text.\n")
            continue
        
        cleaned_text = clean_text(user_text)
        prediction = classifier.classify(cleaned_text)

        print(f"Result: {prediction}\n")


if __name__ == "__main__":
    main()