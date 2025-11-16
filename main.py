# main.py
# ----------------------------------------------------------
# Project: AmbedkarGPT-Intern-Task
# Description: Local Q&A system using LangChain + ChromaDB + Ollama (Mistral 7B)
# ----------------------------------------------------------

import os
import sys

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def check_file_exists(filepath: str):
    if not os.path.exists(filepath):
        print(f"❌ Error: Required file '{filepath}' not found.")
        sys.exit(1)


def initialize_ollama():
    print("🔹 Checking Ollama installation...")
    try:
        os.system("ollama list > nul 2>&1" if os.name == "nt" else "ollama list > /dev/null 2>&1")
    except Exception:
        print("⚠️ Ollama not found. Install it from https://ollama.com")
        sys.exit(1)

    print("🔹 Verifying Mistral model...")
    result = os.system("ollama list | findstr mistral > nul 2>&1" if os.name == "nt"
                       else "ollama list | grep mistral > /dev/null 2>&1")

    if result != 0:
        print("⚠️ Mistral model not found. Run this:")
        print("   ollama pull mistral")
        sys.exit(1)
    else:
        print("✅ Mistral model is installed and working!")


def build_qa_system():
    print("🚀 Starting AmbedkarGPT System Setup...\n")

    check_file_exists("speech.txt")

    print("📘 Loading and splitting text...")
    loader = TextLoader("speech.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("🧠 Creating text embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir = "./chroma_store"
    print(f"💾 Building vector DB at {persist_dir} ...")
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

    initialize_ollama()

    print("🤖 Initializing Mistral 7B ...")
    llm = Ollama(model="mistral")

    print("🔧 Building Retrieval-QA Chain...")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are AmbedkarGPT AI.

Use ONLY the following context to answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and simply.
""",
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )

    print("\n✅ System READY! Type your questions below.\n")

    while True:
        query = input("🧠 Ask: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting... Bye!")
            break

        if len(query) < 3:
            print("⚠️ Question too short.")
            continue

        print("🔍 Thinking...\n")

        try:
            result = qa.invoke({"query": query})
            answer = result["result"]
            print(f"💬 Answer: {answer}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")


if __name__ == "__main__":
    build_qa_system()
