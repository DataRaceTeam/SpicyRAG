import logging
from pydoc import locate

from tqdm import tqdm

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

from interface.chunker import AbstractBaseChunker
from interface.database import SessionLocal
from interface.embedder import Embedder
from interface.models import DataChunks, HmaoNpaDataset, RagasNpaDataset
from interface.schemas import EmbedderSettings

logger = logging.getLogger(__name__)


def initialize_llm_client(config):
    """
    Initializes and returns the LLM client using provided configuration.
    """
    try:
        llm_client = OpenAI(
            base_url=config["llm"]["base_url"], api_key=config["llm"]["api_key"]
        )
        logger.info("LLM client initialized successfully.")
        return llm_client
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        raise


def initialize_embedding_model(config: dict) -> Embedder:
    """
    Initializes and returns the embedding model using provided configuration.
    """
    try:
        settings = EmbedderSettings(**config["embedding_model"])
        logger.info(f"Loaded embedder settings for model {settings.model_name}")
        embedder = Embedder(settings)
        logger.info(f"Initialized embedding model {settings.model_type}, {settings.model_name}")
        return embedder
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise


def chunker(text, max_length, chunk_overlap=256):
    """
    Splits the given text into smaller chunks with an optional overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents([Document(text)])


def load_data(embedder, config):
    """
    Loads and processes data into the database by chunking text and vectorizing it.
    Only loads data into a table if the table is empty.
    """
    db = SessionLocal()
    try:
        # Check if there is any data in the RagasNpaDataset table
        if db.query(RagasNpaDataset).first() is not None:
            logger.info(
                "Data already exists in the RagasNpaDataset table. Skipping data loading for this table."
            )
        else:
            logger.info(
                "No existing data found in RagasNpaDataset. Proceeding with data loading for this table."
            )
            load_excel_data(db, config)

        # Check if there is any data in the HmaoNpaDataset table
        if db.query(HmaoNpaDataset).first() is not None:
            logger.info(
                "Data already exists in the HmaoNpaDataset table. Skipping data loading for this table."
            )
        else:
            logger.info(
                "No existing data found in HmaoNpaDataset. Proceeding with data loading and processing for this table."
            )
            load_and_process_text_documents(db, embedder, config)

        logger.info("Data loading process completed successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def load_excel_data(db, config):
    """
    Loads data from an Excel file into the database.
    """
    try:
        file_path = config["data_sources"]["excel_file"]
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            db_entry = RagasNpaDataset(
                question=row["question"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                evolution_type=row["evolution_type"],
                meta_data=row["metadata"],
                episode_done=row["episode_done"],
            )
            db.add(db_entry)
        db.commit()
        logger.info(f"Loaded data from {file_path}")
    except Exception as e:
        logger.error(f"Error loading data from Excel: {e}")
        raise


def load_and_process_text_documents(db, embedder, config):
    """
    Loads and processes text documents from a file, chunking and vectorizing the content.
    """

    chunker_cls = locate(config["data_processing"]["chunker"]["py_class"])
    chunker: AbstractBaseChunker = chunker_cls(**config["data_processing"]["chunker"]["kwargs"])

    logger.info("Successfully initialized chunker")
    text_transformers = [
        c["py_class"](**c.get("kwargs"))
        for c in config["data_processing"].get("text_transformers", [])
    ]
    chunk_transformers = [
        c["py_class"](**c.get("kwargs"))
        for c in config["data_processing"].get("chunk_transformers", [])
    ]

    try:
        file_path = config["data_sources"]["text_file"]
        summ_path = config["data_sources"]["summ_file"]
        separator = config["data_sources"]["text_separator"]

        with open(file_path, "r", encoding="utf-8") as npa, open(summ_path, "r", encoding="utf-8") as npa_summ:
            content = npa.read().split(separator)

        logger.info("Started vectorizing the NPA data and store it in database")
        for document in tqdm(content):
            hmao_entry = HmaoNpaDataset(document_text=document.strip())
            db.add(hmao_entry)
            db.commit()

            text = hmao_entry.document_text
            for transformer in text_transformers:
                text = transformer.transform(text)

            chunks = chunker.chunk(text)
            for transformer in chunk_transformers:
                chunks = transformer.transform(chunks)

            store_chunks(db, hmao_entry.id, chunks, embedder, config)
        logger.info(f"Processed and stored chunks from {file_path}")
    except Exception as e:
        logger.error(f"Error processing text documents: {e}")
        raise


def store_chunks(db, parent_id, chunks, embedder, config) -> None:
    """
    Vectorizes and stores text chunks in the database.
    """
    try:
        embeddings = embedder.encode(chunks, doc_type="document")

        for passage, embedding in zip(chunks, embeddings, strict=True):
            db_chunk = DataChunks(parent_id=parent_id, chunk_text=passage, vector=embedding)
            db.add(db_chunk)
        db.commit()
    except Exception as e:
        logger.error(f"Error storing chunks in the database: {e}")
        raise


def retrieve_contexts(query, embedder, config):
    """
    Retrieves the most relevant contexts from DataChunks for a given query using vector search.
    """
    db = SessionLocal()
    try:
        # Encode the query to get its vector
        query_vector = embedder.encode([query], doc_type="query")[0].tolist()

        # Define parameters
        k = config["data_processing"]["top_k"]
        similarity_threshold = config["data_processing"]["similarity_threshold"]

        # Query the database for the most similar contexts based on cosine similarity
        results = (
            db.query(
                DataChunks,
                DataChunks.vector.cosine_distance(query_vector).label("distance"),
            )
            .filter(
                DataChunks.vector.cosine_distance(query_vector) < similarity_threshold
            )
            .order_by("distance")
            .limit(k)
            .all()
        )

        # Extract the chunk_texts from the results
        top_chunks = [result.DataChunks.chunk_text for result in results]

        logger.info(f"Retrieved top {k} contexts for the query")
        return top_chunks
    except Exception as e:
        logger.error(f"Error retrieving contexts: {e}")
        raise
    finally:
        db.close()


def generate_response(llm_client, contexts, query, config):
    """
    Generates a response based on retrieved contexts and the input query.
    """
    try:
        prompt = build_prompt(contexts, query)
        response = llm_client.chat.completions.create(
            model=config["llm"]["model"],
            messages=[
                {"role": "system", "content": config["llm"]["system_prompt"]},
                {"role": config["llm"]["role"], "content": prompt},
            ],
            temperature=config["llm"]["temperature"],
            top_p=config["llm"]["top_p"],
            max_tokens=config["llm"]["max_tokens"],
            stream=True,
        )

        generated_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                generated_response += chunk.choices[0].delta.content

        logger.info(f"Generated response: {generated_response[:30]}...")
        return generated_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise


def build_prompt(contexts, query):
    """
    Constructs the prompt for the LLM based on the given contexts and query.
    """

    prompt = "Отвечай используя контекст:\n"
    for i, context in enumerate(contexts):
        prompt += f"Контекст {i + 1}: {context}\n"
    prompt += f"Вопрос: {query}\nНе упоминай, что ты пользуешься контекстом\nПодробный Ответ: "
    return prompt


def rewrite_query(llm_client, query, config):
    """
    Rewrites user's query using LLM_rewriter
    """
    try:
        response = llm_client.chat.completions.create(
            model=config["llm_respond"]["model"],
            messages=[
                {"role": "system", "content": config["llm_respond"]["system_prompt"]},
                {"role": config["llm_respond"]["role"], "content": f"Вопрос: {query}"},
            ],
            temperature=config["llm_respond"]["temperature"],
            top_p=config["llm_respond"]["top_p"],
            max_tokens=config["llm_respond"]["max_tokens"],
            stream=True,
        )

        rewrited_query = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                rewrited_query += chunk.choices[0].delta.content
        
        rewrited_query += f'\n------------------\n{query}'
        logger.info(f"Rewrited query: {rewrited_query}")
        return rewrited_query
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        raise


def process_request(config, llm_client, query):
    """
    Processes the incoming query by retrieving relevant contexts and generating a response.
    """
    try:
        embedder = initialize_embedding_model(config)
        rewrited_query = rewrite_query(llm_client, query, config)
        contexts = retrieve_contexts(
            rewrited_query, embedder, config
        )  # В ретривер отправляется переписанный запрос

        # Generate the response
        llm_response = generate_response(llm_client, contexts, query, config)

        # Return both the response and the contexts used
        return {"response": llm_response, "context": contexts}

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        return (
            "An error occurred while processing your request. Please try again later."
        )
