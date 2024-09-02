import pandas as pd
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from interface.database import SessionLocal
from interface.models import RagasNpaDataset, HmaoNpaDataset, DataChunks

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


def initialize_embedding_model(config):
    """
    Initializes and returns the embedding model using provided configuration.
    """
    try:
        model = SentenceTransformer(config["embedding_model"]["name"])
        logger.info(f"Initialized embedding model {config['embedding_model']['name']}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise


def chunker(text, max_length, overlap_percentage=0.2):
    """
    Splits the given text into smaller chunks with an optional overlap.
    """
    words = text.split()
    overlap_length = int(max_length * overlap_percentage)
    chunks = []

    for i in range(0, len(words), max_length - overlap_length):
        chunk = " ".join(words[i : i + max_length])
        chunks.append(chunk)

    return chunks


def load_data(model, config):
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
            load_and_process_text_documents(db, model, config)

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


def load_and_process_text_documents(db, model, config):
    """
    Loads and processes text documents from a file, chunking and vectorizing the content.
    """
    try:
        file_path = config["data_sources"]["text_file"]
        separator = config["data_sources"]["text_separator"]
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().split(separator)

        for document in content:
            hmao_entry = HmaoNpaDataset(document_text=document.strip())
            db.add(hmao_entry)
            db.commit()

            chunks = chunker(
                hmao_entry.document_text, config["data_processing"]["chunk_size"]
            )
            store_chunks(db, hmao_entry.id, chunks, model, config)
        logger.info(f"Processed and stored chunks from {file_path}")
    except Exception as e:
        logger.error(f"Error processing text documents: {e}")
        raise


def store_chunks(db, parent_id, chunks, model, config):
    """
    Vectorizes and stores text chunks in the database.
    """
    try:
        for chunk in chunks:
            vector = vectorize(chunk, model)
            db_chunk = DataChunks(parent_id=parent_id, chunk_text=chunk, vector=vector)
            db.add(db_chunk)
        db.commit()
    except Exception as e:
        logger.error(f"Error storing chunks in the database: {e}")
        raise


def vectorize(chunk, model):
    """
    Vectorizes a given text chunk using the provided model.
    """
    try:
        vector = model.encode(chunk)
        print("type ", type(vector))
        return vector.tolist()
    except Exception as e:
        logger.error(f"Failed to vectorize chunk: {e}")
        raise


def retrieve_contexts(query, model, config):
    """
    Retrieves the most relevant contexts from DataChunks for a given query using vector search.
    """
    db = SessionLocal()
    try:
        # Encode the query to get its vector
        query_vector = model.encode(query).tolist()

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

        logger.info(f"Retrieved top {k} contexts for the query: {query[:30]}")
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
            messages=[{"role": config["llm"]["role"], "content": prompt}],
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
    prompt = "Ты являешься опытным и знающим ассистентом. На основе приведённых ниже контекстов, ответь на следующий вопрос:\n"
    for i, context in enumerate(contexts):
        prompt += f"Context {i + 1}: {context}\n"
    prompt += f"Question: {query}\nAnswer:"
    return prompt


def process_request(config, llm_client, query):
    """
    Processes the incoming query by retrieving relevant contexts and generating a response.
    """
    try:
        model = initialize_embedding_model(config)
        contexts = retrieve_contexts(query, model, config)

        # Generate the response
        llm_response = generate_response(llm_client, contexts, query, config)

        # Return both the response and the contexts used
        return {"llm_response": llm_response, "contexts": contexts}

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        return (
            "An error occurred while processing your request. Please try again later."
        )
