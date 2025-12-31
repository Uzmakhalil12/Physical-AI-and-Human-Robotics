from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"

class Message(BaseModel):
    message_id: str
    role: MessageRole
    content: str
    timestamp: datetime
    context_used: Optional[List[str]] = []

class ChatSession(BaseModel):
    session_id: str
    messages: List[Message] = []
    created_at: datetime
    updated_at: datetime

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    context_used: List[str]
    is_in_scope: bool