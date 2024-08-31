from fastapi import FastAPI
from interface.database import engine
from interface import utils, models, schemas
import yaml
import logging
from openai import OpenAI

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

# Fetch llm settings from config
openai_base_url = config["llm"]["base_url"]
openai_api_key = config["llm"]["api_key"]
openai_model = config["llm"]["model"]
openai_role = config["llm"]["role"]
openai_temperature = config["llm"]["temperature"]
openai_top_p = config["llm"]["top_p"]
openai_max_tokens = config["llm"]["max_tokens"]

# Initialize llm Client
client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)


@app.post("/ask/")
def ask_question(question: schemas.QuestionCreate):
    logger.info(f"Received question: {question.text}")
    word_count = utils.count_words(question.text)
    logger.info(f"Word count: {word_count}")

    completion = client.chat.completions.create(
        model=openai_model,
        messages=[{"role": openai_role, "content": question.text}],
        temperature=openai_temperature,
        top_p=openai_top_p,
        max_tokens=openai_max_tokens,
        stream=True,
    )

    response_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_content += chunk.choices[0].delta.content

    logger.info(f"LLM Response: {response_content}")

    return {"word_count": word_count, "llm_response": response_content}
