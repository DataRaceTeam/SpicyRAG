import logging
from contextlib import asynccontextmanager

import yaml
from elasticsearch import Elasticsearch
from fastapi import FastAPI

from interface import models, schemas, utils
from interface.database import engine

# Load configuration
config_path = "interface/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize logger
logging.basicConfig(level=logging.getLevelName(config["logging"]["level"]))
logger = logging.getLogger(config["project"]["name"])
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)

# Initialize the database
models.Base.metadata.create_all(bind=engine)

# Initialize models and LLM client
local_embedder = utils.initialize_embedding_model(config)
llm_client = utils.initialize_llm_client(config)

es_client = Elasticsearch(([{"host": config["elastic_params"]["host"], "port": config["elastic_params"]["port"]}]))

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting data loading process.")
        utils.load_data(local_embedder, es_client, config)  # Pass config to load_data
        logger.info("Data loading completed.")
    except Exception as e:
        logger.warning(f"Service starts without any data in the DB caused by: {e}")

    yield
    logger.info("App shutting down")


app = FastAPI(title=config["project"]["name"], lifespan=lifespan)


@app.post("/ask/", response_model=schemas.QuestionResponse)
def ask_question(question: schemas.QuestionCreate):
    logger.info(f"Received question: {question.question}")

    response_content = utils.process_request(config, local_embedder, llm_client, question.question, es_client)

    logger.info(f"LLM Response: {response_content}")

    response = schemas.QuestionResponse(
        response=response_content['response'],
        context=response_content['context']
    )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,
                host=config["server"]["host"],
                port=config["server"]["port"],
                reload=True,
                )
