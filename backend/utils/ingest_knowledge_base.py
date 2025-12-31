import json
import os
from dotenv import load_dotenv
from services.vector_db import VectorDBService

# Load environment variables
load_dotenv()

def ingest_knowledge_base():
    """
    Load the QA dataset and ingest it into the vector database
    """
    print("Starting knowledge base ingestion...")

    # Load the QA dataset
    data_path = os.path.join(os.path.dirname(__file__), "../../data/qa_dataset.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data["qa_pairs"]
    print(f"Loaded {len(qa_pairs)} Q&A pairs from dataset")

    # Initialize the vector database service
    vector_db = VectorDBService()

    # Add each Q&A pair to the vector database
    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair["question"]
        answer = qa_pair["answer"]

        # Add to vector database
        entry_id = vector_db.add_knowledge_entry(
            question=question,
            answer=answer,
            metadata={"source": "physical_ai_robotics_dataset", "category": "general"}
        )

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(qa_pairs)} Q&A pairs")

    print(f"Successfully ingested {len(qa_pairs)} Q&A pairs into the vector database")
    print("Knowledge base ingestion completed!")

if __name__ == "__main__":
    ingest_knowledge_base()