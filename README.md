
# 🔎 Retrieval-Augmented Generation (RAG) — LangChain + Pinecone + Groq

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge&logo=python)]
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)]
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blueviolet?style=for-the-badge)]
[![Groq](https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge)]
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)]

---

## 📌 Project Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline using:

- **LangChain** for orchestration
- **Pinecone** as the vector database
- **HuggingFace embeddings** for semantic encoding
- **Groq (LLaMA 3.1)** for text generation

The system enhances LLM responses by grounding them in external documents stored in a vector database, reducing hallucinations and improving factual accuracy.

---

## 🏗 System Architecture

Document  
↓  
Chunking  
↓  
Embeddings (HuggingFace)  
↓  
Pinecone Vector Store  

User Question  
↓  
Embedding  
↓  
Similarity Search  
↓  
Retrieved Context  
↓  
Groq LLM  
↓  
Grounded Response  

---

## 🧠 Core Components

### 1️⃣ Indexing (`ingest.py`)

- Loads documents
- Splits them into chunks
- Generates embeddings
- Stores vectors in Pinecone

### 2️⃣ Retrieval + Generation (`query.py`)

- Converts user question to embedding
- Retrieves most relevant chunks
- Injects context into prompt
- Generates grounded answer via Groq

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.11
- pip
- Groq API key
- Pinecone API key

---

## ⚙️ Installation

```bash
git clone https://github.com/JeissonS02/rag-project.git
cd rag-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-index
```

---

## ▶️ Running the Project

### 1️⃣ Index Documents

```bash
python ingest.py
```

### 2️⃣ Query the System

```bash
python query.py
```

---

## 📁 Project Structure

```
├── data/
│   └── sample.txt
├── ingest.py
├── query.py
├── requirements.txt
├── .gitignore
├── .env.example
├── .env (not tracked)
└── README.md
```

---

## 📈 Learning Outcomes

- Understanding semantic search with vector databases
- Implementing end-to-end RAG pipelines
- Integrating embeddings with Pinecone
- Grounding LLM responses with retrieved context
- Managing reproducible ML environments

---

## 👨‍💻 Author

[![GitHub](https://img.shields.io/badge/GitHub-JeissonS02-181717?style=for-the-badge&logo=github)](https://github.com/JeissonS02)
