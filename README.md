# Emotion Analysis Engine  
Hybrid ML + Retrieval + LLM System

This project implements a **hybrid emotion analysis pipeline** combining:
- a classical machine learning classifier,
- sentence-embedding–based retrieval,
- and a small controlled language model (LLM).

Instead of returning only an emotion label, the system produces a **grounded,
neutral explanation** of the detected emotion.

---

## What It Does

Given a user sentence, the system:
1. predicts an emotion using a classical ML model,
2. retrieves relevant scientific context,
3. generates a factual, non-conversational explanation.

---

## High-Level Architecture

- **Classifier:** TF-IDF + Logistic Regression  
- **Retrieval:** sentence-transformers (Mini-RAG)  
- **LLM:** TinyLlama-1.1B for explanation generation  
- **Interface:** CLI application  

---

## Scope & Limitations

- Fixed emotion set (6 classes)
- Baseline classifier, not production-grade
- Small LLM with limited instruction following
- No MLOps, monitoring, or scalable serving

These trade-offs are intentional and aligned with the project’s learning scope.

---

## Documentation

📘 **Full technical documentation:**  
👉 [README.full.md](README.full.md)


---

## Quick Start

```bash
python -m src.chatbot_interface
```

---

*This is a learning and portfolio project, focused on clarity,
explainability, and engineering structure rather than production readiness.*
