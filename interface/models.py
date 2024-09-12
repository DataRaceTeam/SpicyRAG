import yaml
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
)

from interface.database import Base

# Load configuration to access vector dimension
config_path = "interface/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


class RagasNpaDataset(Base):
    __tablename__ = "ragas_npa_dataset"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    contexts = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    evolution_type = Column(String, nullable=False)
    meta_data = Column(JSON, nullable=False)
    episode_done = Column(Boolean, nullable=False)


class HmaoNpaDataset(Base):
    __tablename__ = "hmao_npa_dataset"

    id = Column(Integer, primary_key=True, index=True)
    dt = Column(Date, nullable=False)
    npa_number = Column(String, nullable=False)
    document_text = Column(Text, nullable=False)


class DataChunks(Base):
    __tablename__ = "data_chunks"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("hmao_npa_dataset.id"), nullable=False)
    dt = Column(Date, nullable=False)
    npa_number = Column(String, nullable=False)
    chunk_text = Column(Text, nullable=False)
    vector = Column(Vector(config["embedding_model"]["dimension"]))

    def __init__(self, parent_id, dt, npa_number, chunk_text, vector):
        self.parent_id = parent_id
        self.dt = dt
        self.npa_number = npa_number
        self.chunk_text = chunk_text
        self.vector = vector
