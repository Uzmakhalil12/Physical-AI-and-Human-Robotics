import qdrant_client
from qdrant_client import models
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

class VectorDBService:
    def __init__(self):
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url:
            raise ValueError("QDRANT_URL environment variable is required")
        if not api_key:
            raise ValueError("QDRANT_API_KEY environment variable is required")

        self.client = qdrant_client.QdrantClient(
            url=url,
            api_key=api_key,
        )
        self.collection_name = "knowledge_base"
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """
        Create the knowledge base collection if it doesn't exist
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection if it doesn't exist
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                )
        except Exception as e:
            # If collection already exists, ignore the error
            if "already exists" not in str(e).lower():
                raise

    def add_knowledge_entry(self, question: str, answer: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a knowledge base entry with its embedding
        """
        from services.embeddings import EmbeddingService
        embedding_service = EmbeddingService()

        # Create embedding for the question
        embedding = embedding_service.embed_text(question)

        # Create a unique ID for this entry
        entry_id = str(uuid.uuid4())

        # Prepare the point for Qdrant
        point = models.PointStruct(
            id=entry_id,
            vector=embedding,
            payload={
                "question": question,
                "answer": answer,
                "metadata": metadata or {}
            }
        )

        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        return entry_id

    def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar knowledge base entries to the query
        """
        from services.embeddings import EmbeddingService
        embedding_service = EmbeddingService()

        # Create embedding for the query
        query_embedding = embedding_service.embed_text(query)

        # Search in Qdrant using modern API
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )

        # Format results
        formatted_results = []
        for result in results.points:
            formatted_results.append({
                "id": result.id,
                "question": result.payload["question"],
                "answer": result.payload["answer"],
                "metadata": result.payload.get("metadata", {}),
                "score": result.score
            })

        return formatted_results

    def get_all_entries(self) -> List[Dict]:
        """
        Get all knowledge base entries (for debugging/inspection)
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            limit=10000  # Adjust as needed
        )

        formatted_results = []
        for point, _ in results:
            formatted_results.append({
                "id": point.id,
                "question": point.payload["question"],
                "answer": point.payload["answer"],
                "metadata": point.payload.get("metadata", {})
            })

        return formatted_results