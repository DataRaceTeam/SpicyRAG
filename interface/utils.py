import pandas as pd
from interface.database import SessionLocal
from interface.models import RagasNpaDataset, HmaoNpaDataset, DataChunks
from sentence_transformers import SentenceTransformer


# Function to chunk text into smaller parts
def chunker(text, max_length=200):
    words = text.split()
    chunks = [
        " ".join(words[i : i + max_length]) for i in range(0, len(words), max_length)
    ]
    return chunks


# Initialize the local model
def initialize_local_model():
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Example smaller model
    return model


# Function to vectorize a text chunk using a local model
def vectorize(chunk, model):
    vector = model.encode(chunk)
    return vector.tolist()  # Convert numpy array to list for storing in the database


def load_data(model):
    db = SessionLocal()

    try:
        file_path = "./data/v2_ragas_npa_dataset_firstPart.xlsx"
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

        file_path = "./data/hmao_npa.txt"
        separator = "\n\n"
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().split(separator)

        for document in content:
            hmao_entry = HmaoNpaDataset(document_text=document.strip())
            db.add(hmao_entry)
            db.commit()

            chunks = chunker(hmao_entry.document_text)
            for chunk in chunks:
                vector = vectorize(chunk, model)
                db_chunk = DataChunks(
                    parent_id=hmao_entry.id, chunk_text=chunk, vector=vector
                )
                db.add(db_chunk)
        db.commit()

    finally:
        db.close()


def process_request(config, llm_client, question):
    llm_model = config["llm"]["model"]
    llm_role = config["llm"]["role"]
    llm_temperature = config["llm"]["temperature"]
    llm_top_p = config["llm"]["top_p"]
    llm_max_tokens = config["llm"]["max_tokens"]

    completion = llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": llm_role, "content": question.text}],
        temperature=llm_temperature,
        top_p=llm_top_p,
        max_tokens=llm_max_tokens,
        stream=True,
    )

    response_content = ""

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_content += chunk.choices[0].delta.content
