from fastapi import FastAPI
from contextlib import asynccontextmanager
from interface.database import engine
from interface import utils, models, schemas
import yaml
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load configuration
config_path = "interface/config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize the database
models.Base.metadata.create_all(bind=engine)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(config["project"]["name"])

# Initialize the local embedding model
local_embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Initialize LLM Client
llm_base_url = config["llm"]["base_url"]
llm_api_key = config["llm"]["api_key"]

llm_client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting data loading process.")
        utils.load_data(local_embedding_model)  # Pass the local model to load_data
        logger.info("Data loading completed.")
    except Exception as e:
        logger.warning(f"Service starts without any data in the DB caused by: {e}")

    yield
    logger.info("App shutting down")


app = FastAPI(title=config["project"]["name"], lifespan=lifespan)


@app.post("/ask/")
def ask_question(question: schemas.QuestionCreate):
    logger.info(f"Received question: {question.text}")
    response_content = utils.process_request(config, llm_client, question.text)

    logger.info(f"LLM Response: {response_content}")
    return {"llm_response": response_content}
