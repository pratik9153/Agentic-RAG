"""Streamlit UI for Agentic RAG."""

import sys
import time
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).parent

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# YOUR PROJECT IMPORTS
# ============================================================

from src.config.config import Config
from src.document_ingestion.document_process import DocumentProcess
from src.vectorestore.vecterstore import VectorStore
from src.graph_builder.graph import GraphBuilder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "chats" not in st.session_state:
        st.session_state.chats = {
            "New Chat": []
        }

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "New Chat"


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

@st.cache_resource
def initialize_rag():

    # Create LLM
    llm = Config.get_llm()

    # Create document processor
    document_processor = DocumentProcess(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    # PDF path
    pdf_path = (
        PROJECT_ROOT
        / "data"
        / "ilovepdf_merged.pdf"
    )

    # Check PDF
    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Load PDF
    documents = document_processor.load_pdf(
        str(pdf_path)
    )

    # Split PDF into chunks
    chunks = document_processor.split_documents(
        documents
    )

    # Create vector store
    vector_store = VectorStore(
        chunks
    )

    # Create retriever
    vector_store.create_retriever()

    # Get retriever
    retriever = vector_store.get_retriever()

    # Create graph
    graph_builder = GraphBuilder(
        retriever=retriever,
        llm=llm
    )

    # Build graph
    graph_builder.build()

    return graph_builder


# ============================================================
# CREATE NEW CHAT
# ============================================================

def create_new_chat():

    chat_number = len(
        st.session_state.chats
    )

    chat_name = f"Chat {chat_number}"

    st.session_state.chats[chat_name] = []

    st.session_state.current_chat = chat_name


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.title("🤖 Agentic RAG")

        st.caption(
            "AI-powered document assistant"
        )

        st.divider()

        # New chat button
        if st.button(
            "➕ New Chat",
            use_container_width=True
        ):

            create_new_chat()

            st.rerun()

        st.markdown("### 💬 Chat History")

        # Display chats
        for chat_name in st.session_state.chats:

            if st.button(
                chat_name,
                key=f"chat_{chat_name}",
                use_container_width=True
            ):

                st.session_state.current_chat = (
                    chat_name
                )

                st.rerun()

        st.divider()

        # Model information
        st.markdown("### 🧠 Model")

        st.caption(
            Config.MODEL_NAME
        )

        # Document information
        st.markdown("### 📄 Document")

        st.caption(
            "ilovepdf_merged.pdf"
        )


# ============================================================
# WELCOME SCREEN
# ============================================================

def show_welcome():

    st.title("🤖 Agentic RAG")

    st.subheader(
        "Intelligent Document Assistant"
    )

    st.write(
        "Ask questions about your PDF and let the "
        "AI agent decide which tool to use."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "📄 Document Search"
        )

        st.write(
            "Search information from your uploaded "
            "PDF using semantic retrieval."
        )

    with col2:

        st.subheader(
            "🧠 Agentic Reasoning"
        )

        st.write(
            "The agent decides which available "
            "tool should be used."
        )

    with col3:

        st.subheader(
            "🔎 Contextual Answers"
        )

        st.write(
            "Generate answers using retrieved "
            "information and relevant context."
        )

    st.divider()

    st.info(
        "💡 Try asking: What is this document about?"
    )


# ============================================================
# DISPLAY CHAT
# ============================================================

def display_chat(messages):

    for message in messages:

        role = message["role"]

        content = message["content"]

        with st.chat_message(role):

            st.markdown(content)

            # Tool information
            tool = message.get("tool")

            if tool:

                st.caption(
                    f"🔧 Tool: {tool}"
                )

            # Sources
            sources = message.get(
                "sources",
                []
            )

            if sources:

                with st.expander(
                    f"📄 Sources ({len(sources)})"
                ):

                    for index, document in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"**Source {index}**"
                        )

                        st.text(
                            document.page_content[:1000]
                        )

            # Response time
            response_time = message.get(
                "time"
            )

            if response_time:

                st.caption(
                    f"⏱️ Response time: "
                    f"{response_time:.2f} seconds"
                )


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(
    question,
    messages
):

    # Save user question
    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user question
    with st.chat_message("user"):

        st.markdown(question)

    # Check RAG system
    if st.session_state.rag_system is None:

        st.error(
            "RAG system is not initialized."
        )

        return

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Agent is thinking..."
        ):

            start_time = time.time()

            result = (
                st.session_state
                .rag_system
                .run(question)
            )

            elapsed_time = (
                time.time()
                - start_time
            )

        # Get answer
        answer = result.get(
            "answer",
            "Could not generate an answer."
        )

        st.markdown(answer)

        # Get sources
        sources = result.get(
            "retrieved_docs",
            []
        )

        if sources:

            with st.expander(
                f"📄 Sources ({len(sources)})"
            ):

                for index, document in enumerate(
                    sources,
                    start=1
                ):

                    st.markdown(
                        f"**Source {index}**"
                    )

                    st.text(
                        document.page_content[:1000]
                    )

        # Response time
        st.caption(
            f"⏱️ Response time: "
            f"{elapsed_time:.2f} seconds"
        )

        # Save assistant message
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "time": elapsed_time,
                "tool": "Agentic RAG"
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # Initialize session
    initialize_session_state()

    # Show sidebar
    show_sidebar()

    # Initialize RAG system
    if not st.session_state.initialized:

        with st.spinner(
            "🚀 Loading Agentic RAG..."
        ):

            try:

                rag_system = initialize_rag()

                st.session_state.rag_system = (
                    rag_system
                )

                st.session_state.initialized = True

            except Exception as error:

                st.error(
                    f"Failed to initialize system: {error}"
                )

                return

    # Get current chat
    messages = st.session_state.chats[
        st.session_state.current_chat
    ]

    # Show welcome screen
    if not messages:

        show_welcome()

    # Show conversation
    else:

        display_chat(messages)

    # Chat input
    question = st.chat_input(
        "Ask something about your documents..."
    )

    # Process question
    if question:

        process_question(
            question,
            messages
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()