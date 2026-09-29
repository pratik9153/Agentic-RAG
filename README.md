# 🤖 Agentic RAG – Intelligent Document Assistant

An **Agentic RAG application** that allows users to ask questions about a PDF.
The AI agent decides which tool to use and generates a relevant answer.

## 🚀 Features

* 📄 Ask questions about uploaded PDF documents
* 🔎 Semantic search using vector embeddings
* 🤖 AI agent for tool selection
* 🌐 Wikipedia search for general knowledge
* 🧠 Agentic RAG using LangGraph
* 💬 Chat interface using Streamlit
* 📚 Source document support
* ⚡ Response time tracking
* 💬 Multiple chat sessions

## 🏗️ Architecture

```text
                    User Question
                          │
                          ▼
                     🤖 AI Agent
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
     📄 Document Search          🌐 Wikipedia Search
             │                         │
             ▼                         ▼
        PDF + FAISS               Wikipedia API
             │                         │
             └────────────┬────────────┘
                          ▼
                    Final Answer
                          │
                          ▼
                   Streamlit UI
```

## 🛠️ Tech Stack

| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python        | Main programming language |
| LangChain     | LLM and tool integration  |
| LangGraph     | Agent workflow            |
| OpenAI        | LLM and embeddings        |
| FAISS         | Vector similarity search  |
| Pydantic      | State management          |
| Streamlit     | User interface            |
| Wikipedia API | General knowledge search  |

## 📂 Project Structure

```text
RAG Project/
│
├── data/
│   └── ilovepdf_merged.pdf
│
├── src/
│   │
│   ├── config/
│   │   └── config.py
│   │
│   ├── document_ingestion/
│   │   └── document_process.py
│   │
│   ├── vectorestore/
│   │   └── vecterstore.py
│   │
│   ├── graph_builder/
│   │   └── graph.py
│   │
│   ├── nodes/
│   │   └── nodes.py
│   │
│   └── state/
│       └── rag_state.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## 🔄 How It Works

### 1. Document Loading

The application loads the PDF using the document processing module.

```text
PDF
 ↓
Document Loader
 ↓
Documents
```

### 2. Text Splitting

The documents are divided into smaller chunks.

```text
Documents
 ↓
Text Splitter
 ↓
Document Chunks
```

### 3. Vector Store

The chunks are converted into embeddings and stored in **FAISS**.

```text
Document Chunks
 ↓
Embeddings
 ↓
FAISS Vector Store
```

### 4. Agent

The user question is passed to the AI agent.

The agent decides which tool should be used:

```text
Question
   │
   ▼
 AI Agent
   │
   ├── PDF related → Document Search
   │
   └── General knowledge → Wikipedia Search
```

### 5. Final Answer

The selected tool provides information to the agent, and the agent generates the final response.

## 🧰 Available Tools

### 📄 Document Search

Searches the uploaded PDF using the FAISS retriever.

Used for questions related to the uploaded document.

### 🌐 Wikipedia Search

Uses the Wikipedia API to search for general information.

Used when the question is not specifically related to the uploaded PDF.

## 💻 Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the Project

```bash
cd RAG-Project
```

### 3. Create Environment

Using `uv`:

```bash
uv venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
uv add -r requirements.txt
```

If you face a Windows hardlink issue:

```bash
set UV_LINK_MODE=copy
```

Then run:

```bash
uv add -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

Do not upload your API key to GitHub.

Add `.env` to `.gitignore`:

```text
.env
.venv/
__pycache__/
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
uv run streamlit run streamlit_app.py
```

The application will open in your browser.

## 💡 Example Questions

### PDF Question

```text
What is this document about?
```

### General Knowledge Question

```text
What is machine learning?
```

### Another General Question

```text
What are the types of machine learning?
```

The agent decides whether to search the PDF or Wikipedia.

## 🎯 What I Learned

Through this project, I worked with:

* RAG pipelines
* Vector databases
* Semantic search
* LangChain
* LangGraph
* AI agents
* Agentic RAG
* Tool calling
* API integration
* Pydantic state
* Streamlit application development

## 🔮 Future Improvements

* Add conversation memory
* Add more tools
* Add query rewriting
* Add hybrid search
* Add RAG evaluation
* Add LangSmith observability
* Add authentication
* Deploy the application to the cloud

## 👨‍💻 Project

**Agentic RAG – Intelligent Document Assistant**

Built using **Python, LangChain, LangGraph, OpenAI, FAISS and Streamlit**.

---

⭐ If you find this project useful, consider giving it a star.
