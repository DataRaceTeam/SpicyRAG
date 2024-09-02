import requests
from sqlalchemy.orm import Session
from database import SessionLocal
from models import RagasNpaDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    SummarizationScore,
)
from ragas import evaluate
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize RAGAS metrics
metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall(),
    SummarizationScore(),
]


def load_test_data(db: Session):
    """
    Load questions and ground truth answers from RagasNpaDataset.
    """
    logger.info("Loading test data from RagasNpaDataset")
    return db.query(RagasNpaDataset).all()


def get_system_responses(questions):
    """
    Send requests to the /ask endpoint and collect responses.
    """
    responses = []
    for question in questions:
        logger.info(f"Sending question to /ask endpoint: {question.question}")
        response = requests.post(
            "http://interface:8000/ask/", json={"text": question.question}
        )
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Received response: {response_data}")
        responses.append(
            {
                "question": question.question,
                "response": response_data["llm_response"],
                "ground_truth": question.ground_truth,
                "context": response_data["contexts"],
            }
        )
    return responses


def test_evaluate_system():
    """
    Evaluate the system's responses using the RAGAS framework.
    """
    db = SessionLocal()
    try:
        # Load the test data
        test_data = load_test_data(db)

        # Send questions to the /ask endpoint and collect responses
        system_responses = get_system_responses(test_data)

        # Prepare data for RAGAS evaluation
        predictions = [response["response"] for response in system_responses]
        references = [response["ground_truth"] for response in system_responses]
        contexts = [response["context"] for response in system_responses]

        # Evaluate the system using RAGAS
        logger.info("Evaluating system responses using RAGAS")
        results = evaluate(predictions, references, contexts, metrics=metrics)

        # Assertions for each metric to ensure tests pass if scores are above a threshold
        for metric_name, score in results.items():
            logger.info(f"{metric_name}: {score:.2f}")
            assert score > 0.01, f"{metric_name} score is too low: {score}"
    finally:
        db.close()
        logger.info("Database connection closed")
