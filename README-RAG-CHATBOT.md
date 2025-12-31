# Physical AI and Human Robotics RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot specifically designed to answer questions about Physical AI and Human Robotics. The system uses Cohere's embedding and generation models with Qdrant vector database to provide accurate, domain-specific responses.

## Architecture

The system consists of:
- **Backend**: FastAPI application handling chat queries and RAG processing
- **Frontend**: React chatbot component integrated into the Docusaurus website
- **Vector Database**: Qdrant for storing and retrieving knowledge base embeddings
- **AI Models**: Cohere embed-english-v3.0 for embeddings and command-r for generation

## Features

- Domain-specific responses limited to Physical AI & Human Robotics topics
- Out-of-scope query detection with appropriate responses
- Session management for maintaining conversation context
- Responsive chat interface
- Knowledge base with 100+ Q&A pairs

## Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- Cohere API key
- Qdrant API key and URL

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. The `.env` file is already in the backend directory with your API keys.

4. Ingest the knowledge base into the vector database:
   ```bash
   python utils/ingest_knowledge_base.py
   ```

5. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

The frontend is integrated into the existing Docusaurus website in the frontend directory. To run it:

1. Navigate to the frontend directory and install dependencies:
   ```bash
   cd frontend
   yarn
   ```

2. Start the development server:
   ```bash
   yarn start
   ```

## Usage

1. Access the chatbot at `/chatbot` on your Docusaurus site
2. Ask questions related to Physical AI & Human Robotics
3. The chatbot will retrieve relevant information from the knowledge base and generate responses
4. Out-of-scope questions will receive a specific response asking for relevant questions

## API Endpoints

- `POST /api/chat/query` - Process a user query using RAG
- `GET /api/chat/session/{session_id}` - Retrieve a chat session
- `DELETE /api/chat/session/{session_id}` - Delete a chat session

## Environment Variables

- `COHERE_API_KEY` - Your Cohere API key
- `QDRANT_URL` - Your Qdrant instance URL
- `QDRANT_API_KEY` - Your Qdrant API key

## Data Pipeline

The system uses a pre-populated knowledge base with over 100 Q&A pairs related to Physical AI and Human Robotics. The `utils/ingest_knowledge_base.py` script processes this data and creates embeddings for efficient retrieval.

## Security

- All API keys are loaded from environment variables
- Input validation is performed on all user queries
- Scope validation ensures only domain-relevant questions are processed