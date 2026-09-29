from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStore:

    def __init__(self, documents: List[Document]):

        self.documents = documents

        self.embedding = OpenAIEmbeddings()

        self.vector_store = None
        self.retriever = None

    # Create FAISS vector store and retriever
    def create_retriever(self):

        self.vector_store = FAISS.from_documents(
            self.documents,
            self.embedding
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

    # Return retriever
    def get_retriever(self):

        if self.retriever is None:
            raise ValueError("Vector store not initialized")

        return self.retriever

    # Search documents using query
    def retrieve(self, query: str) -> List[Document]:

        if self.retriever is None:
            raise ValueError("Retriever not initialized")

        return self.retriever.invoke(query)