# Quickstart Guide for Physical AI and Human Robotics RAG Chatbot

## Development Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker (for containerized deployment)
- Cohere API key
- Qdrant API key and URL

### Environment Variables
Create a `.env` file with the following:
```
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Backend Setup
1. Install Python dependencies: `pip install fastapi uvicorn cohere qdrant-client python-dotenv`
2. Set up environment variables
3. Run the backend: `uvicorn main:app --reload`

### Frontend Setup
1. Navigate to frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start development server: `npm start`

## API Usage

### Query Endpoint
```
POST /api/chat/query
Content-Type: application/json

{
  "query": "What is Physical AI?",
  "session_id": "optional-session-id"
}
```

### Session Management
```
GET /api/chat/session/{session_id}
DELETE /api/chat/session/{session_id}
```

## Data Pipeline

### Ingest Knowledge Base
Run the ingestion script to convert QA dataset to vector embeddings:
```
python scripts/ingest_knowledge_base.py
```

## Testing
Run backend tests: `pytest tests/`
Run frontend tests: `npm test`

## Deployment
1. Build Docker images for backend and frontend
2. Configure environment variables for production
3. Deploy containers to your preferred platform