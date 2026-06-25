# Multi-Tool RAG Chatbot with LangGraph

This repository contains a multi-tool RAG (Retrieval-Augmented Generation) chatbot application. The system handles document ingestion, autonomous tool routing, persistent conversation states, and session recovery.

---

## Features

* **PDF Ingestion & RAG**: Upload documents via the UI. The system splits the text into chunks using a recursive character splitter and indexes them into a local FAISS vector store using HuggingFace embeddings for precise similarity matching.
* **Persistent Memory & Session Recovery**: Utilizes an automated SQLite backend (`chatbot.db`) via LangGraph's `SqliteSaver`. Conversation histories are preserved across sessions and can be fully resumed using distinct thread IDs.
* **Autonomous Tool Routing**: Built on top of a modular `StateGraph`. The LLM (Llama 3.3 via Groq) dynamically evaluates user inputs to determine if it should use internal knowledge or trigger a specific tool.
* **Integrated Tool Ecosystem**:
  * **Document Retriever (`rag_tool`)**: Contextually queries the indexed PDF associated with the active chat thread.
  * **Math Engine (`calculator`)**: Executes precise arithmetic operations (addition, subtraction, multiplication, division) to prevent LLM calculation errors.
  * **Web Search (`search_tool`)**: Queries DuckDuckGo for real-time information retrieval when requested context falls outside the indexed document.
* **Observability**: Designed with strict, decoupled state tracking nodes (`chat_node` and `tools`), allowing step-by-step auditing of the agent's decision-making flow.

---

## Repository Structure

* **`backend.py`**: Houses the LangGraph workflow compilation, tool definitions, embedding/vector store management, and the SQLite checkpointer database connection.
* **`frontend.py`**: Implements the Streamlit user interface, handling UI chat loops, historical session renders, and file upload stream transfers to the backend.

---

## Quick Start

### 1. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq credentials:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Install Required Packages
Install the application dependencies via your terminal:
```bash
pip install streamlit langchain langchain-groq langchain-huggingface langgraph faiss-cpu pypdf python-dotenv
```

### 3. Run the Application
Launch the local web server using Streamlit:
```bash
streamlit run frontend.py
```
