import logging
from contextlib import asynccontextmanager

import envyaml
from fastapi import FastAPI

from interface import models, schemas, utils
from interface.database import engine

# Load configuration
config = envyaml.EnvYAML("interface/config.yaml")

# Initialize logger
logging.basicConfig(level=logging.getLevelName(config["logging"]["level"]))
logger = logging.getLogger(config["project"]["name"])

# Initialize the database
models.Base.metadata.create_all(bind=engine)

# Initialize models and LLM client
local_embedding_model = utils.initialize_embedding_model(config)
llm_client = utils.initialize_llm_client(config)

reranker = utils.initialize_reranker(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting data loading process.")
        utils.load_data(local_embedding_model, config)  # Pass config to load_data
        logger.info("Data loading completed.")
    except Exception as e:
        logger.warning(f"Service starts without any data in the DB caused by: {e}")

    yield
    logger.info("App shutting down")


app = FastAPI(title=config["project"]["name"], lifespan=lifespan)


@app.post("/ask/")
def ask_question(question: schemas.QuestionCreate):
    logger.info(f"Received question: {question.question}")
    response_content = utils.process_request(config, llm_client, reranker, question.question)

    logger.info(f"LLM Response: {response_content}")
    return response_content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,
                host=config["server"]["host"],
                port=config["server"]["port"],
                reload=True,
                )
