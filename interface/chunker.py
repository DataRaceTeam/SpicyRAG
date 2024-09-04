from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
            separators=separators
        )

    def chunk(self, text: str) -> list[str]:
        return [d.page_content for d in self.text_splitter.split_documents([Document(text)])]


class AbstractTransformChunk(ABC):

    @abstractmethod
    def transform(self, chunks: list[str], text: str) -> list[str]:
        pass


class AddHeaderTransformChunk(AbstractTransformChunk):
    """Add header from document to chunk"""

    def transform(self, chunks: list[str], text: str) -> list[str]:
        header = text[:[i for i in range(len(text)) if text[i] == '"'][1] + 1]

        return [header + c for c in chunks]
