# Physical AI and Human Robotics RAG Chatbot Constitution

## Core Principles

### I. Domain Scope Adherence
All responses must be related to Physical AI & Human Robotics. Any out-of-scope questions must be replied with: "This question is outside the scope of Physical AI and Human Robotics. Please ask a relevant question." This ensures the chatbot maintains its specialized expertise and doesn't provide potentially inaccurate information on unrelated topics.

### II. Accuracy Over Guessing
The chatbot must not hallucinate answers. All responses must be generated using retrieval-augmented generation (RAG) from the embedded knowledge base. This principle ensures factual accuracy and prevents the spread of misinformation.

### III. Technology Stack Standardization
Backend: FastAPI for API endpoints. Frontend: React for chatbot interface. Embeddings: Cohere embed-english-v3.0. Generation: Cohere command-r. Vector DB: Qdrant free tier. This standardizes the technology stack for maintainability and optimal performance.

### IV. Knowledge Source Integrity
All chatbot answers must come exclusively from the knowledge base. The system must implement proper RAG retrieval logic to ensure responses are grounded in the provided data. This maintains consistency and reliability.

### V. Error Handling and User Experience
The system must provide graceful error handling for API failures, network issues, and invalid inputs. User experience must remain smooth even during partial system failures, with appropriate fallback messages.

### VI. Performance and Efficiency
API endpoints must respond within reasonable timeframes (under 5 seconds for typical queries). The system should efficiently manage resources and implement proper caching where appropriate to optimize performance.

## Technical Requirements

### Backend Architecture
- FastAPI application with proper routing for chat endpoints
- Environment variable configuration for API keys and service URLs
- Proper error handling and logging
- Input validation and sanitization

### Frontend Integration
- React component for chat interface
- Real-time message display
- Loading states and error handling
- Responsive design for various screen sizes

### Data Management
- Q&A dataset preparation for embedding
- Vector database integration with Qdrant
- Proper indexing and search capabilities
- Data backup and recovery procedures

## Development Workflow

### Code Quality Standards
- Type hints for all function signatures
- Comprehensive error handling
- Proper documentation for all public interfaces
- Unit tests for critical functionality
- Code follows PEP 8 standards for Python and standard React patterns

### Review Process
- All code changes must pass automated tests
- Peer review required for all pull requests
- Performance testing for API endpoints
- Security review for any new dependencies

## Governance

The constitution governs all development practices for this project. All changes to this constitution require explicit approval and must be documented with rationale. All PRs and reviews must verify compliance with these principles. Use this constitution as the primary guidance for development decisions.

**Version**: 1.0.0 | **Ratified**: 2025-12-28 | **Last Amended**: 2025-12-28
