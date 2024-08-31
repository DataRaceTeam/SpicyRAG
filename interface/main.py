from fastapi import FastAPI
from interface.database import engine
from interface import utils, models, schemas
import yaml
import logging

# Load configuration
config_path = "interface/config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Initialize the database
models.Base.metadata.create_all(bind=engine)

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(config["project"]["name"])

app = FastAPI(title=config["project"]["name"])


@app.post("/ask/")
def ask_question(question: schemas.QuestionCreate):
    logger.info(f"Received question: {question.text}")
    word_count = utils.count_words(question.text)
    logger.info(f"Word count: {word_count}")
    return {"word_count": word_count}
