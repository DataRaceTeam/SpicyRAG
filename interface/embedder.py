import abc
from typing import List

import more_itertools
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, XLMRobertaTokenizer

from interface.schemas import EmbedderSettings


class IEmbedder(abc.ABC):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @abc.abstractmethod
    def encode(self, sentences: List[str], doc_type: str) -> np.ndarray:
        """Calculate sentences embedding(s)"""


class Embedder(IEmbedder):
    def __init__(self, settings: EmbedderSettings):
        super().__init__()
        self._settings = settings
        self.batch_size = self._settings.batch_size
        self.model_type = self._settings.model_type
        self.prefix_query = self._settings.prefix_query
        self.prefix_document = self._settings.prefix_document

        if self.model_type == 'e5':
            self.model = XLMRobertaModel.from_pretrained(self._settings.model_name).to(self.device)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self._settings.model_name)
        else:
            self.model = AutoModel.from_pretrained(self._settings.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self._settings.model_name)

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, sentences: List[str], doc_type: str) -> np.ndarray:
        sentences = self.preprocess_sentences(sentences, doc_type)
        embeddings = torch.tensor([]).to(self.device)

        for batch in tqdm(more_itertools.chunked(sentences, self.batch_size)):
            tokenized_batch = self.tokenizer(batch, max_length=512, padding=True,
                                             truncation=True, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokenized_batch).last_hidden_state
                embed = self.average_pool(outputs, tokenized_batch['attention_mask'])

            torch.cuda.empty_cache()

            for tensor in embed:
                embeddings = torch.cat((embeddings, tensor.unsqueeze(0)), 0)

        return np.array([torch.Tensor.cpu(emb) for emb in F.normalize(embeddings, dim=-1)])

    def preprocess_sentences(self, sentences: List[str], doc_type: str) -> List[str]:
        if doc_type == 'query':
            return [self.prefix_query.format(sentence) for sentence in sentences]
        elif doc_type == 'document':
            return [self.prefix_document.format(sentence) for sentence in sentences]
        return sentences
