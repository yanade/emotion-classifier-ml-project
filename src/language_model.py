from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.config import LANGUAGE_MODEL_NAME
from src.simple_interface import clean_input
from src.classifier import Classifier
from src.retriever import Retriever

class Chatbot:
    model_name = LANGUAGE_MODEL_NAME

    def __init__(self, classifier: Classifier, retriever: Retriever):
        self.model_name = LANGUAGE_MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Chatbot is using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.system_prompt = ("<|system|>\n"
                                "You are a emotional explanation assistant.\n"
                                "When given an emotion label and context, explain ONLY that emotion.\n"
                                "Do NOT confuse it with other emotions.\n"
                                "<|end|>\n")

        self.classifier = classifier
        self.retriever = retriever
        print(f"Model requested: {self.model_name}")
        print(f"Device used: {self.device}")

    def build_prompt(self, new_user_message: str) -> str:
        """Builds a fresh, stateless prompt for the current query."""
        prompt = self.system_prompt
        prompt += f"<|user|>\n{new_user_message}\n<|end|>\n<|assistant|>\n"
        return prompt

    def encode_prompt(self, prompt: str):
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        return encoded

    def generate_reply(self, text: str) -> str:
        prompt = self.build_prompt(text)
        encoded = self.encode_prompt(prompt)
     
        generated_output = self.model.generate(**encoded,
                                                   pad_token_id=self.tokenizer.eos_token_id,
                                                   do_sample=True,
                                                   max_new_tokens=60,
                                                   temperature=0.9,
                                                   top_p=0.9,
                                                   top_k=500)
        
        decoded = self.tokenizer.decode(generated_output[0], skip_special_tokens=False)
       
        try:
            assistant_text = decoded.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
        except:
            assistant_text = decoded.strip()

        return assistant_text
        
    def classify_with_llm(self, user_text: str):
        cleaned_message = clean_input(user_text)
        detected_emotion = self.classifier.classify(cleaned_message)
        retrieved_chunks_knowledge = self.retriever.retrieve(cleaned_message, top_k=1)
        context_chunk_knowledge = "\n".join(f" - {chunk}" for chunk in retrieved_chunks_knowledge)
        
        llm_instruction = (
            f"The user said: \"{user_text}\"\n\n"
            f"Detected emotion: {detected_emotion}\n"
            f"Relevant scientific notes:\n{context_chunk_knowledge}\n\n"
            "Task:\nExplain the detected emotion clearly and neutrally, based   "
            "on the scientific notes.\n"
            "Do NOT mention other emotions.\n"
            "Do NOT address the user directly or use an empathetic tone.\n"
        )
       
        reply = self.generate_reply(llm_instruction)
        return detected_emotion, reply