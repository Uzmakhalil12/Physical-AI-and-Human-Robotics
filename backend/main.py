from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from typing import Optional, List
import os
from dotenv import load_dotenv

# Import our modules
from api.chat import router as chat_router
from models.session import ChatSession

# Load environment variables
load_dotenv()

# Create the main FastAPI app
app = FastAPI(
    title="Physical AI and Human Robotics RAG Chatbot API",
    description="API for the Physical AI and Human Robotics RAG Chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the chat router
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Physical AI and Human Robotics RAG Chatbot API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}