"""Configuration for Agentic RAG system."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI




load_dotenv()


class Config:
    """Central configuration for the Agentic RAG system."""

    

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    

    MODEL_NAME = "gpt-4o-mini"

    TEMPERATURE = 0


   
   

    CHUNK_SIZE = 500

    CHUNK_OVERLAP = 50


    
   


   

    @classmethod
    def get_llm(cls):
        """Create and return the ChatOpenAI model."""

        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file."
            )

        return ChatOpenAI(
            model=cls.MODEL_NAME,
            temperature=cls.TEMPERATURE,
            api_key=cls.OPENAI_API_KEY
        )