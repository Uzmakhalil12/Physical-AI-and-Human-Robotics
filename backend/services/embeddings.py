from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingService:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # OpenRouter uses OpenAI-compatible SDK
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        # Using a model that supports dimensions parameter to match existing data (1024)
        # If text-embedding-3-small is used, we can specify dimensions=1024
        self.model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text
        """
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            extra_body={"dimensions": 1024} if "text-embedding-3" in self.model else {}
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            extra_body={"dimensions": 1024} if "text-embedding-3" in self.model else {}
        )
        return [item.embedding for item in response.data]
