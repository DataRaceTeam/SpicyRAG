from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Dict
import re


class AbstractBaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass


class RecursiveCharacterTextSplitterChunker(AbstractBaseChunker):
    def __init__(self, chunk_size, chunk_overlap, separators=None):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            is_separator_regex=False,
            separators=separators,
        )

    def chunk(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)


class SemanticTextChunker(AbstractBaseChunker):
    def __init__(
        self,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=None,
        number_of_chunks=None,
    ):
        self.splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
        )

    def chunk(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


class AbstractTransformChunk(ABC):
    @abstractmethod
    def transform(self, chunks: list[str], text: str) -> list[str]:
        pass


class AddHeaderTransformChunk(AbstractTransformChunk):
    """Add header from document to chunk"""

    def transform(self, chunks: list[str], text: str) -> list[str]:
        header = text[: [i for i in range(len(text)) if text[i] == '"'][1] + 1]

        return [header + c for c in chunks]


class ColBERTChunker:
    def __init__(self, max_length: int = 512, overlap: int = 50):
        self.max_length = max_length
        self.overlap = overlap

    def chunk(self, text: str) -> List[Dict[str, str]]:
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_header = ""

        for para in paragraphs:
            # Check if this paragraph is a header
            if len(para.split()) < 10 and para.strip().endswith(':'):
                current_header = para.strip()
                continue

            # If adding this paragraph would exceed max_length, store the current chunk and start a new one
            if len(current_chunk) + len(para) > self.max_length:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "header": current_header})
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

            # If the current chunk is long enough, store it and start a new one with overlap
            if len(current_chunk) >= self.max_length - self.overlap:
                chunks.append({"text": current_chunk.strip(), "header": current_header})
                words = current_chunk.split()
                current_chunk = " ".join(words[-self.overlap:])

        # Add any remaining text as a final chunk
        if current_chunk:
            chunks.append({"text": current_chunk.strip(), "header": current_header})

        return chunks