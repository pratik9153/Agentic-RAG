"""Nodes and tools for the Agentic RAG workflow."""

from typing import List

import requests

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent

from src.state.rag_state import RAGState


class RAGNodes:

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm

        self.agent = None

    # --------------------------------------------------
    # Create tools
    # --------------------------------------------------
    def create_tools(self):

        retriever = self.retriever

        # --------------------------------------------------
        # Tool 1: Search uploaded PDF
        # --------------------------------------------------
        @tool
        def document_search(query: str) -> str:
            """
            Search the uploaded PDF for relevant information.
            Use this tool when the user's question is related
            to information that may exist in the uploaded PDF.
            """

            documents: List[Document] = (
                retriever.invoke(query)
            )

            if not documents:
                return (
                    "No relevant information was found "
                    "in the uploaded PDF."
                )

            results = []

            for index, document in enumerate(
                documents,
                start=1
            ):

                results.append(
                    f"Document {index}:\n"
                    f"{document.page_content}"
                )

            return "\n\n".join(results)

        # --------------------------------------------------
        # Tool 2: Wikipedia Search
        # --------------------------------------------------
        @tool
        def wikipedia_search(query: str) -> str:
            """
            Search Wikipedia for general knowledge.
            Use this tool when the question is not specifically
            about the uploaded PDF.
            """

            url = "https://en.wikipedia.org/w/api.php"

            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": 3
            }

            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=10,
                    headers={
                        "User-Agent": "AgenticRAG/1.0"
                    }
                )

                response.raise_for_status()

                data = response.json()

            except requests.RequestException as error:

                return (
                    f"Wikipedia search failed: {error}"
                )

            except ValueError:

                return (
                    "Wikipedia returned an invalid response."
                )

            search_results = (
                data
                .get("query", {})
                .get("search", [])
            )

            if not search_results:

                return (
                    "No relevant Wikipedia results found."
                )

            results = []

            for index, item in enumerate(
                search_results,
                start=1
            ):

                title = item.get(
                    "title",
                    ""
                )

                snippet = item.get(
                    "snippet",
                    ""
                )

                results.append(
                    f"Result {index}: {title}\n"
                    f"{snippet}"
                )

            return "\n\n".join(results)

        return [
            document_search,
            wikipedia_search
        ]

    # --------------------------------------------------
    # Create Agent
    # --------------------------------------------------
    def create_agent(self):

        tools = self.create_tools()

        system_prompt = """
You are an Agentic RAG assistant.

You have two tools:

1. document_search
   - Searches the user's uploaded PDF.
   - Use this when the question is related to
     information that may exist in the PDF.

2. wikipedia_search
   - Searches Wikipedia.
   - Use this for general knowledge questions.

Rules:

- Decide which tool is appropriate for the question.

- For questions about the uploaded document,
  use document_search.

- For general knowledge questions,
  use wikipedia_search.

- Always use an appropriate tool before answering.

- You may use a tool more than once if necessary.

- After receiving the tool result, use that
  information to produce the final answer.

- Do not expose your internal reasoning.

- If a tool does not provide useful information,
  clearly say that the information could not be found.

Give a clear and concise final answer.
"""

        self.agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=system_prompt
        )

        return self.agent

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------
    def generate_answer(
        self,
        state: RAGState
    ) -> dict:

        # Create agent only once
        if self.agent is None:

            self.create_agent()

        # Run Agent
        result = self.agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=state.question
                    )
                ]
            }
        )

        messages = result.get(
            "messages",
            []
        )

        answer = ""

        # Find final AI response
        for message in reversed(messages):

            content = getattr(
                message,
                "content",
                ""
            )

            if isinstance(content, str):

                if content.strip():

                    answer = content

                    break

        if not answer:

            answer = (
                "I could not generate an answer."
            )

        return {
            "answer": answer
        }