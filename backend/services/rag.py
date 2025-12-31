from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv
from .embeddings import EmbeddingService
from .vector_db import VectorDBService

# Load environment variables
load_dotenv()

class RAGService:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # OpenRouter uses OpenAI-compatible SDK
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite-preview-02-05:free")
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

        # Prepare the system message/prompt
        system_prompt = f"Based on the following context from the Physical AI and Human Robotics knowledge base, please answer the user's question.\n\nContext:\n{context}\n\nPlease provide a helpful and accurate answer based on the context provided. If the context doesn't contain the information needed to answer the question, please say so."

        # Generate response using OpenAI-compatible Chat Completion API via OpenRouter
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            max_tokens=500,
            temperature=0.3,
        )

        generated_text = response.choices[0].message.content.strip()

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

        return False
