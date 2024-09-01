import requests
from sqlalchemy.orm import Session
from interface.database import SessionLocal
from interface.models import RagasNpaDataset
from ragas.metrics import (
    Faithfulness, AnswerRelevance, ContextPrecision,
    ContextRecall, SummarizationScore
)
from ragas import Ragas

# Initialize RAGAS metrics
metrics = [
    Faithfulness(),
    AnswerRelevance(),
    ContextPrecision(),
    ContextRecall(),
    SummarizationScore()
]

# Initialize RAGAS evaluator
ragas_evaluator = Ragas(metrics)


# Function to load questions and ground truth answers from RagasNpaDataset
def load_test_data(db: Session):
    return db.query(RagasNpaDataset).all()


# Function to send requests to the /ask endpoint and collect responses
def get_system_responses(questions):
    responses = []
    for question in questions:
        response = requests.post(
            "http://interface:8000/ask/",
            json={"text": question.question}
        )
        response.raise_for_status()
        response_data = response.json()
        responses.append({
            "question": question.question,
            "response": response_data["llm_response"],
            "ground_truth": question.ground_truth,
            "context": question.contexts
        })
    return responses


def test_evaluate_system():
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
        results = ragas_evaluator.evaluate(predictions, references, contexts)

        # Assertions for each metric to ensure tests pass if scores are above a threshold
        for metric_name, score in results.items():
            print(f"{metric_name}: {score:.2f}")
            assert score > 0.01, f"{metric_name} score is too low: {score}"
    finally:
        db.close()
