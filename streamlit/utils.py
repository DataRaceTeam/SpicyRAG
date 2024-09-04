import requests
from ragas import Ragas
from ragas.metrics import (
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    SummarizationScore,
)

# Initialize RAGAS metrics
metrics = [
    Faithfulness(),
    AnswerRelevance(),
    ContextPrecision(),
    ContextRecall(),
    SummarizationScore(),
]

# Initialize RAGAS evaluator
ragas_evaluator = Ragas(metrics)


def get_system_responses(questions):
    """Send requests to the /ask endpoint and collect responses."""
    responses = []
    for question in questions:
        response = requests.post(
            "http://interface:8000/ask/", json={"text": question.question}
        )
        response_data = response.json()
        responses.append(
            {
                "question": question.question,
                "response": response_data["llm_response"],
                "ground_truth": question.ground_truth,
                "context": question.contexts,
            }
        )
    return responses
