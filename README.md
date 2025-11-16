# AmbedkarGPT-Intern-Task

## 🎯 Project Overview
This is the AI Intern Assignment (Phase 1 - Core Skills Evaluation) for Kalpit Pvt Ltd, UK.

You are building a command-line Q&A system that:
- Reads Dr. B.R. Ambedkar’s short speech (`speech.txt`)
- Creates embeddings using HuggingFace
- Stores them in ChromaDB
- Uses Ollama (Mistral 7B) to answer questions based on that content

---

## ⚙️ Tech Stack
- **Language:** Python 3.8+
- **Framework:** LangChain
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **LLM:** Ollama (`mistral`)

---

## 🧩 Installation Steps

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # (Mac/Linux)
venv\Scripts\activate       # (Windows)
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Install & Setup Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
```

### 4️⃣ Run the Program
```bash
python main.py
```

---

## 💬 Example Interaction
```
🧠 Ask a question: What is the main problem mentioned by Ambedkar?
💬 Answer: The problem of caste, which comes from belief in the sanctity of the shastras.
```

## 📁 File Structure
```
AmbedkarGPT-Intern-Task/
│
├── main.py
├── requirements.txt
├── README.md
└── speech.txt
```

## 👨‍💻 Author
Kalpit Pvt Ltd - AI Intern Assignment (UK)
kalpiksingh2005@gmail.com
