import pandas as pd
from sqlalchemy.orm import Session
from interface.database import SessionLocal, engine
from interface.models import RagasNpaDataset, HmaoNpaDataset, Base


# Function to load Excel data into RagasNpaDataset
def load_excel_data(db: Session):
    file_path = "./data/v2_ragas_npa_dataset_firstPart.xlsx"
    df = pd.read_excel(file_path)

    for _, row in df.iterrows():
        db_entry = RagasNpaDataset(
            question=row["question"],
            contexts=row["contexts"],
            ground_truth=row["ground_truth"],
            evolution_type=row["evolution_type"],
            meta_data=row[
                "metadata"
            ],  # Make sure to adjust this field name in the DataFrame if needed
            episode_done=row["episode_done"],
        )
        db.add(db_entry)
    db.commit()


# Function to load text data into HmaoNpaDataset
def load_text_data(db: Session):
    file_path = "./data/hmao_npa.txt"
    separator = "\n\n"  # Using double newlines as the separator

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().split(separator)

    for document in content:
        db_entry = HmaoNpaDataset(document_text=document.strip())
        db.add(db_entry)
    db.commit()


if __name__ == "__main__":
    db = SessionLocal()

    Base.metadata.create_all(engine)

    try:
        load_excel_data(db)
        load_text_data(db)
    finally:
        db.close()
