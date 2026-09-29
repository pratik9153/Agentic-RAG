"""Build the LangGraph workflow for Agentic RAG."""

from langgraph.graph import StateGraph, END

from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:

    def __init__(self, retriever, llm):

        # Create RAG nodes
        self.nodes = RAGNodes(
            retriever,
            llm
        )

        self.graph = None

    def build(self):

        # Create LangGraph
        builder = StateGraph(RAGState)

        # Add Agent node
        builder.add_node(
            "agent",
            self.nodes.generate_answer
        )

        # Start with Agent
        builder.set_entry_point(
            "agent"
        )

        # Agent finishes
        builder.add_edge(
            "agent",
            END
        )

        # Compile graph
        self.graph = builder.compile()

        return self.graph

    def run(self, question: str):

        # Create initial state
        state = RAGState(
            question=question
        )

        # Run Agentic RAG
        result = self.graph.invoke(
            state
        )

        # Return final answer
        return {
            "answer": result.get(
                "answer",
                "I do not know the answer."
            ),
            "retrieved_docs": result.get(
                "retrieved_docs",
                []
            )
        }