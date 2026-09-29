"""LangGraph nodes for Agentic RAG workflow."""

from typing import List, Optional

from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains nodes for the Agentic RAG workflow."""

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm

        # Agent will be created only when needed
        self._agent = None

    
    # Build tools
    
    def _build_tools(self) -> List[Tool]:
        """Create tools available to the ReAct agent."""

       
        # PDF Retriever Tool
       
        def retriever_tool_fn(query: str) -> str:

            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No relevant information found in the uploaded documents."

            results = []

            for i, document in enumerate(docs[:8], start=1):

                metadata = document.metadata

                source = (
                    metadata.get("title")
                    or metadata.get("source")
                    or f"Document {i}"
                )

                results.append(
                    f"[Document {i}] {source}\n"
                    f"{document.page_content}"
                )

            return "\n\n".join(results)

        retriever_tool = Tool(
            name="document_search",
            description=(
                "Search the user's uploaded PDF/document collection. "
                "Use this tool when the question is related to information "
                "that may exist in the user's documents."
            ),
            func=retriever_tool_fn,
        )

        
        # Wikipedia Tool
        
        wikipedia_api = WikipediaAPIWrapper(
            top_k_results=3,
            lang="en"
        )

        wikipedia_search = WikipediaQueryRun(
            api_wrapper=wikipedia_api
        )

        wikipedia_tool = Tool(
            name="wikipedia_search",
            description=(
                "Search Wikipedia for general knowledge. "
                "Use this when the required information is not available "
                "in the user's documents or when general knowledge is needed."
            ),
            func=wikipedia_search.run,
        )

        return [
            retriever_tool,
            wikipedia_tool
        ]

    
    # Build ReAct Agent
    
    def _build_agent(self):
        """Create the ReAct agent with available tools."""

        tools = self._build_tools()

        system_prompt = """
You are an Agentic RAG assistant.

You have access to two tools:

1. document_search
   - Searches the user's uploaded documents/PDF.
   - Prefer this tool when the question is related to the uploaded documents.

2. wikipedia_search
   - Searches Wikipedia for general knowledge.
   - Use this when the required information is not available in the documents
     or when the question requires general knowledge.

Decide which tool is appropriate for each question.

You may use a tool more than once if necessary.

After collecting enough information, provide a clear and useful final answer.
Do not explain your internal reasoning.
"""

        self._agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt
        )

        return self._agent

    
   
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate the final answer using the ReAct agent."""

        # Create agent only when required
        if self._agent is None:
            self._build_agent()

        # Send user question to the agent
        result = self._agent.invoke(
            {
                "messages": [
                    HumanMessage(content=state.question)
                ]
            }
        )

        # Get messages returned by the agent
        messages = result.get("messages", [])

        answer: Optional[str] = None

        if messages:

            answer_message = messages[-1]

            answer = getattr(
                answer_message,
                "content",
                None
            )

        # Return updated state
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate an answer."
        )
    
    