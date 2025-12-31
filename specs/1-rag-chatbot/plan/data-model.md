# Data Model for Physical AI and Human Robotics RAG Chatbot

## Entities

### User Query
- **Fields**:
  - query_text: string (required) - The user's input question
  - session_id: string (optional) - Identifier for the chat session
  - timestamp: datetime (auto-generated) - When the query was received
  - validated: boolean (auto-generated) - Whether the query passed scope validation

### Knowledge Base Entry
- **Fields**:
  - id: string (required) - Unique identifier for the knowledge entry
  - question: string (required) - The original question from the dataset
  - answer: string (required) - The original answer from the dataset
  - embedding: array[float] (required) - Vector representation of the content
  - metadata: object (optional) - Additional information like category, source, etc.

### Chat Session
- **Fields**:
  - session_id: string (required) - Unique identifier for the session
  - messages: array[Message] (required) - List of messages in the session
  - created_at: datetime (auto-generated) - When the session was created
  - updated_at: datetime (auto-generated) - When the session was last updated

### Message
- **Fields**:
  - message_id: string (required) - Unique identifier for the message
  - role: string (required) - Either "user" or "assistant"
  - content: string (required) - The text content of the message
  - timestamp: datetime (auto-generated) - When the message was created
  - context_used: array[string] (optional) - IDs of knowledge base entries used to generate the response