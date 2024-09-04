import logging

import requests
import tqdm
from database import SessionLocal
from models import RagasNpaDataset
from sqlalchemy.orm import Session

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_test_data(db: Session, max_cases: int = 1):
    """
    Load up to `max_cases` questions and ground truth answers from RagasNpaDataset.

    Parameters:
    - db: The SQLAlchemy session object.
    - max_cases: The maximum number of cases to load. If None, all cases are loaded.

    Returns:
    - A list of RagasNpaDataset instances.
    """
    logger.info(f"Loading test data from RagasNpaDataset with max_cases={max_cases}")
    query = db.query(RagasNpaDataset)

    if max_cases:
        query = query.limit(max_cases)

    return query.all()


def get_system_responses(questions):
    """
    Send requests to the /ask endpoint and collect responses.
    """
    responses = []
    for question in tqdm.tqdm(questions):
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

        assert system_responses is not None
        assert system_responses["status"] == 200

    finally:
        db.close()
        logger.info("Database connection closed")
