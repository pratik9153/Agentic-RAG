from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List , Union
from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader
)

class DocumentProcess:

    def __init__(self,chunk_size:int=500,chunk_overlap: int=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
     # Load a single PDF file
    def load_pdf(self, file_path: str) -> List[Document]:

        loader = PyMuPDFLoader(file_path)

        documents = loader.load()

        return documents
     # Split PDF into smaller chunks
    def split_documents(self, documents: List[Document]) -> List[Document]:

        chunks = self.splitter.split_documents(documents)

        return chunks
    

    

    



