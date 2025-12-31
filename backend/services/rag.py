import cohere
from typing import List, Dict
import os
from dotenv import load_dotenv
from .embeddings import EmbeddingService
from .vector_db import VectorDBService

# Load environment variables
load_dotenv()

class RAGService:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.cohere_client = cohere.Client(api_key)
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService()

    def query(self, user_query: str, session_id: str = None) -> Dict:
        """
        Process a user query using RAG (Retrieval-Augmented Generation)
        """
        # First, check if the query is in scope
        is_in_scope = self._check_scope(user_query)

        if not is_in_scope:
            return {
                "response": "This question is outside the scope of Physical AI and Human Robotics. Please ask a relevant question.",
                "session_id": session_id or "",
                "context_used": [],
                "is_in_scope": False
            }

        # Search for relevant knowledge base entries
        relevant_entries = self.vector_db_service.search_similar(user_query, limit=3)

        if not relevant_entries:
            return {
                "response": "I couldn't find relevant information in the Physical AI and Human Robotics knowledge base to answer your question.",
                "session_id": session_id or "",
                "context_used": [],
                "is_in_scope": True
            }

        # Prepare context from retrieved entries
        context_texts = []
        context_ids = []
        for entry in relevant_entries:
            context_texts.append(f"Q: {entry['question']}\nA: {entry['answer']}")
            context_ids.append(entry['id'])

        # Combine context and generate response
        context = "\n\n".join(context_texts)

        # Prepare the prompt for Cohere
        prompt = f"""
        Based on the following context from the Physical AI and Human Robotics knowledge base, please answer the user's question.

        Context:
        {context}

        User question: {user_query}

        Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain the information needed to answer the question, please say so.
        """

        # Generate response using Cohere Chat API with current model
        response = self.cohere_client.chat(
            model="command-r-plus-08-2024",
            message=user_query,
            preamble=f"Based on the following context from the Physical AI and Human Robotics knowledge base, please answer the user's question.\n\nContext:\n{context}\n\nPlease provide a helpful and accurate answer based on the context provided. If the context doesn't contain the information needed to answer the question, please say so.",
            max_tokens=500,
            temperature=0.3,
        )

        generated_text = response.text.strip()

        return {
            "response": generated_text,
            "session_id": session_id or "",
            "context_used": context_ids,
            "is_in_scope": True
        }

    def _check_scope(self, query: str) -> bool:
        """
        Check if a query is within the Physical AI & Human Robotics domain
        """
        # This is a simple keyword-based check; in a production system,
        # you might want to use more sophisticated NLP techniques
        query_lower = query.lower()

        # Keywords that indicate the query is in scope
        in_scope_keywords = [
            "physical ai", "physical artificial intelligence", "robotics",
            "human robot interaction", "hri", "embodied ai", "robot",
            "physical interaction", "robot manipulation", "robot control",
            "robot learning", "robot perception", "robot navigation",
            "robotics research", "ai robotics", "robotics ai",
            "physical intelligence", "robotics engineering", "robotics technology"
        ]

        # Check if any in-scope keyword is in the query
        for keyword in in_scope_keywords:
            if keyword in query_lower:
                return True

        # Additional check: if the query seems to be about robotics or AI concepts
        # but doesn't explicitly contain the keywords, we might want to be more permissive
        # For now, we'll use a simple approach - if it's not clearly in scope, it's out of scope
        return False