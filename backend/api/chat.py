from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import uuid
from datetime import datetime
import asyncio
from models.session import QueryRequest, QueryResponse, ChatSession, Message, MessageRole
from services.rag import RAGService

router = APIRouter()

# In-memory storage for sessions (in production, use a proper database)
sessions: Dict[str, ChatSession] = {}

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a user query using RAG
    """
    try:
        # Create or get session
        session_id = request.session_id or str(uuid.uuid4())

        # Initialize session if it doesn't exist
        if session_id not in sessions:
            sessions[session_id] = ChatSession(
                session_id=session_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

        # Add user message to session
        user_message = Message(
            message_id=str(uuid.uuid4()),
            role=MessageRole.user,
            content=request.query,
            timestamp=datetime.utcnow()
        )
        sessions[session_id].messages.append(user_message)

        # Process query with RAG service
        rag_service = RAGService()
        result = rag_service.query(request.query, session_id)

        # Create assistant message
        assistant_message = Message(
            message_id=str(uuid.uuid4()),
            role=MessageRole.assistant,
            content=result["response"],
            timestamp=datetime.utcnow(),
            context_used=result["context_used"]
        )
        sessions[session_id].messages.append(assistant_message)

        # Update session timestamp
        sessions[session_id].updated_at = datetime.utcnow()

        # Return the response
        return QueryResponse(
            response=result["response"],
            session_id=result["session_id"],
            context_used=result["context_used"],
            is_in_scope=result["is_in_scope"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/session/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """
    Retrieve a specific chat session
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return sessions[session_id]


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a specific chat session
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"status": "deleted"}


@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions (for debugging)
    """
    return {"sessions": list(sessions.keys())}