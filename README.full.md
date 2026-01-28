# Emotion Analysis Engine — Full Documentation

This document contains the **complete technical description** of the Emotion Analysis Engine project.

---

## Dataset

This project uses the **dair-ai/emotion** dataset for educational purposes.

**Dataset:**  
https://huggingface.co/datasets/dair-ai/emotion

**Citation:**  
Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018).  
*CARER: Contextualized Affect Representations for Emotion Recognition.*  
EMNLP 2018.  
https://www.aclweb.org/anthology/D18-1404

---

## System Overview

The pipeline consists of three explicit stages:

### 1. Emotion Classification (ML)

- **Model:** TF-IDF + Logistic Regression  
- **Output:** one of six emotions  
  *(sadness, joy, love, anger, fear, surprise)*

The classifier is trained on cleaned text data derived from the dataset.

---

### 2. Context Retrieval (Mini-RAG)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Scientific and definitional texts are embedded offline
- The most relevant sentence is retrieved at inference time

This step provides **factual grounding** for the explanation.

---

### 3. Explanation Generation (LLM)

- **Model:** TinyLlama-1.1B-Chat
- Input to the LLM:
  - predicted emotion label
  - retrieved scientific context
  - strict formatting and tone constraints

The LLM is constrained to produce:
- neutral, factual explanations
- no conversational tone
- no interpretation beyond the classifier output

---

## Assumptions & Limitations

- Fixed emotion taxonomy
- Baseline classifier with limited linguistic robustness
- Sentence-level retrieval only
- Small LLM with limited reasoning capacity
- No experiment tracking, retraining, or deployment automation

This system is **not production-ready by design**.

---

## Tech Stack

- **Language:** Python 3
- **ML & Data:** scikit-learn, pandas, numpy
- **Embeddings & LLMs:** PyTorch, Hugging Face, sentence-transformers
- **Interface:** CLI
- **Testing:** pytest

---

## Project Structure

```
emotion-classifier-ml-project/
├── data/
│   ├── cleaned_dataset.csv
│   └── label_mapping.json
├── knowledge/
│   └── *.txt
├── src/
│   ├── classifier.py
│   ├── retriever.py
│   ├── language_model.py
│   ├── chatbot_interface.py
│   ├── ingest.py
│   ├── train_model.py
│   ├── sentence_model.py
│   ├── model.pkl
│   └── sentence_model.pkl
├── test/
│   └── *.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## First-Time Setup

```bash
python -m src.ingest
python -m src.sentence_model
python -m src.train_model
```

---

## Running the Application

```bash
python -m src.chatbot_interface
```

Example:

```
Enter a sentence: I feel nervous before my exam

Emotion detected: fear
Explanation:
Fear is a biological response associated with activation of stress-regulation
and threat-detection mechanisms in the brain.
```

---

## Testing

The project includes automated tests covering:
- classifier predictions
- retrieval correctness
- explanation generation
- full pipeline integration

Run tests with:

```bash
pytest
```
