from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    JSON,
    Boolean,
    ForeignKey,
    ARRAY,
    Float,
)
from interface.database import Base


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
    document_text = Column(Text, nullable=False)


class DataChunks(Base):
    __tablename__ = "data_chunks"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("hmao_npa_dataset.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    vector = Column(ARRAY(Float), nullable=False)
