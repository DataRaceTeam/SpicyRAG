import logging
from pydoc import locate

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from interface.chunker import AbstractBaseChunker
from interface.database import SessionLocal
from interface.models import DataChunks, HmaoNpaDataset, RagasNpaDataset

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
    Initializes and returns the embedding model and tokenizer using the provided configuration.
    """
    try:
        model_name = config["embedding_model"]["name"]
        model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model {model_name}")
        return model
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
    Loads and processes text documents, chunking and vectorizing the content.
    """

    chunker_cls = locate(config["data_processing"]["chunker"]["py_class"])
    chunker: AbstractBaseChunker = chunker_cls(
        config["data_processing"]["chunker"]["kwargs"]
    )

    text_transformers = [
        c["py_class"](**c.get("kwargs"))
        for c in config["data_processing"]["text_transformers"]
    ]
    chunk_transformers = [
        c["py_class"](**c.get("kwargs"))
        for c in config["data_processing"]["chunk_transformers"]
    ]

    try:
        file_path = config["data_sources"]["text_file"]
        separator = config["data_sources"]["text_separator"]
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().split(separator)

        for document in content:
            hmao_entry = HmaoNpaDataset(document_text=document.strip())
            db.add(hmao_entry)
            db.commit()
            text = hmao_entry.document_text
            for transformer in text_transformers:
                text = transformer.transform(text)

            chunks = chunker.chunk(text)
            for transformer in chunk_transformers:
                chunks = transformer.transform(chunks)

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
            db_chunk = DataChunks(
                parent_id=parent_id, chunk_text=chunk.page_content, vector=vector
            )
            db.add(db_chunk)
        db.commit()
    except Exception as e:
        logger.error(f"Error storing chunks in the database: {e}")
        raise


def vectorize(text, model):
    """
    Vectorizes a given text input using the provided model.
    Accepts a string or Document object and handles the conversion.
    """
    try:
        # TODO: align output / input of classes
        # TODO: align size of chunk with embedding model dim
        # Validate and extract text from Document if necessary
        if isinstance(text, Document):
            logger.info(f"Extracting text from Document type {type(text)}")
            text = text.page_content  # Assuming `Document.page_content` holds the text

        elif not isinstance(text, str):
            raise ValueError("Text input must be of type `str` or `Document`.")

        # Ensure the text is not empty
        if not text.strip():
            raise ValueError("Text input is empty. Cannot vectorize empty text.")

        # Tokenize and vectorize the text
        logger.info(f"Vectorizing text with length {len(text)}")
        embeddings = model.encode(text, convert_to_tensor=True)

        logger.info("Vectorization completed")
        return embeddings.squeeze().tolist()
    except Exception as e:
        logger.error(f"Failed to vectorize text: {e}")
        raise


def retrieve_contexts(query, model, config):
    """
    Retrieves the most relevant contexts from DataChunks for a given query using vector search.
    """
    db = SessionLocal()
    try:
        query_vector = vectorize(query, model)
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
    prompt += f"Вопрос: {query}\nAnswer:"
    return prompt


def rewrite_query(llm_client, query, config):
    """
    Rewrites user's query using LLM_rewriter
    """
    try:
        response = llm_client.chat.completions.create(
            model=config["llm_rewriter"]["model"],
            messages=[
                {"role": "system", "content": config["llm_rewriter"]["system_prompt"]},
                {"role": config["llm_rewriter"]["role"], "content": query},
            ],
            temperature=config["llm_rewriter"]["temperature"],
            top_p=config["llm_rewriter"]["top_p"],
            max_tokens=config["llm_rewriter"]["max_tokens"],
            stream=True,
        )

        rewrited_query = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                rewrited_query += chunk.choices[0].delta.content

        logger.info(f"Rewritten query: {rewrited_query[:30]}...")
        return rewrited_query
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        raise


def process_request(config, llm_client, query):
    """
    Processes the incoming query by retrieving relevant contexts and generating a response.
    """
    try:
        model = initialize_embedding_model(config)
        rewrited_query = rewrite_query(llm_client, query, config)
        contexts = retrieve_contexts(rewrited_query, model, config)

        # Generate the response
        llm_response = generate_response(llm_client, contexts, query, config)

        # Return both the response and the contexts used
        return {"response": llm_response, "context": contexts}

    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        return (
            "An error occurred while processing your request. Please try again later."
        )


# Apply average pooling to model's hidden states
def average_pool(last_hidden_states, attention_mask):
    """
    Applies average pooling to the model's output to create a single embedding vector.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
