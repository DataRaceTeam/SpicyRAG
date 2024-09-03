from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class AbstractBaseChunker(ABC):

    @abstractmethod
    def chunk(self, text):
        pass

    @abstractmethod
    def chunk_texts(self, documents):
        pass


class RecursiveCharacterTextSplitterChunker(AbstractBaseChunker):

    def __init__(self, chunk_size, chunk_overlap, separators=None):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            is_separator_regex=False,
            separators=separators
        )

    def chunk(self, text):
        return self.text_splitter.split_documents([Document(text)])

    def chunk_texts(self, texts):
        return self.text_splitter.split_documents([Document(t) for t in texts])

