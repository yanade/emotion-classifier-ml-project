from src.emotion_classifier import EmotionClassifier
from src.config import MODEL_PATH
from src.language_model import Chatbot
import os

import logging


os.makedirs("logs", exist_ok=True)


logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

bot = Chatbot()

def main():
    print("Welcome! Before we start, what should I call you?")
    user_name = input("Your name: ").strip()
    logging.info(f"User '{user_name}' logged in")

    print(f" Hello {user_name}, I am Emotion classifier - simple interface.")
    print("Type :q or exit to leave.\n")
    print("Tell me something and I'll tell you the emotion.\n")

    exit_conditions = ("q", "exit")
    while True:
        user_input = input("Enter a sentence: ").strip()
        
        if user_input in exit_conditions:
            logging.info(f"User '{user_name}' exited the session")
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter non-empty text.\n")
            continue

        logging.info(f"User input: {user_input}")
        try:
            emotion, response = bot.classify_with_llm(user_input)
            logging.info(f"Emotion classification: {emotion}")
            logging.info(f"LLM response: {response}")
        except Exception as e:
            logging.error(f"Error during classification: {e}")
            print("An error occurred. Please try again.\n")
            continue

        print(f"Emotion: {emotion}")
        print(f"Response: {response}")
    
    

if __name__ == "__main__":
    result = main()
    