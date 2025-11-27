from src.emotion_classifier import EmotionClassifier
from src.config import MODEL_PATH
from src.language_model import Chatbot




bot = Chatbot()

def main():
    print(" Hello, I am Emotion classifier - simple interface.")
    print("Type :q or exit to leave.\n")
    print("Type a sentence to classify.\n")

    exit_conditions = ("q", "exit")
    while True:
        user_input = input("Enter a sentence: ").strip()
        
        if user_input in exit_conditions:
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter non-empty text.\n")
            continue
        emotion, reply = bot.classify_with_llm(user_input)
        print(f"Emotion: {emotion}")
        print(f"Response: {reply}")
    

if __name__ == "__main__":
    result = main()
    