from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.config import LANGUAGE_MODEL_NAME, MODEL_PATH
from src.simple_interface import clean_input
from src.emotion_classifier import EmotionClassifier

class Chatbot:

    model_name = LANGUAGE_MODEL_NAME

    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Chatbot is using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.history: list[tuple[str, str]] = []
        self.system_prompt = "<|system|>\nYou are an emotion-aware helpful assistant.\n<|end|>\n"
        self.classifier = EmotionClassifier(MODEL_PATH)
        print(f"Model requested: {self.model_name}")
        print(f"Device used: {self.device}")

    def build_prompt(self, new_user_message: str) -> str:
        prompt = self.system_prompt
        for user_msg, assistant_msg in self.history: 
            prompt += f"<|user|>\n{user_msg}\n<|end|>\n"
            prompt += f"<|assistant|>\n{assistant_msg}\n<|end|>\n"
        
        prompt += f"<|user|>\n{new_user_message}\n<|end|>\n<|assistant|>\n"
        return prompt


    def encode_prompt(self, prompt: str):
        encoded =self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        return encoded


    def generate_reply(self, prompt: str) -> str:
        user_message = prompt.strip()
        full_prompt = self.build_prompt(user_message)
        
        encoded = self.encode_prompt(full_prompt)
     
        generated_output = self.model.generate(**encoded,
                                                   pad_token_id=self.tokenizer.eos_token_id,
                                                   do_sample=True,
                                                   max_new_tokens=100,
                                                   temperature=0.3,
                                                   top_p=0.9,
                                                   top_k=500)
        
        
        decoded = self.tokenizer.decode(generated_output[0], skip_special_tokens=False)
        assistant_part = decoded.split("<|assistant|>")[-1]
        reply = assistant_part.split("<|end|>")[0].strip()
        
        return reply
    


    def classify_with_llm(self,new_user_message: str):
        cleaned_massage = clean_input(new_user_message)
        detected_emotion = self.classifier.classify(cleaned_massage)
        llm_instruction = (
            f"User: \"{new_user_message}\"\n"
            f"Detected emotion: {detected_emotion}\n"
            "Explain the emotion in a friendly way.\n"
            "Provide empathetic advice and support.\n"
        )
        
        reply = self.generate_reply(llm_instruction)
        self.history.append((new_user_message,reply))
        return detected_emotion, reply