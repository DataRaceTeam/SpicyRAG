import re
from abc import ABC, abstractmethod

import pandas as pd
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseTextTransformation(ABC):

    @abstractmethod
    def transform(self, text: str) -> str:
        pass


class RegexpTextTransformation(BaseTextTransformation):

    def transform(self, text: str) -> str:
        pass


class TfidfFilterTextTransformation(BaseTextTransformation):

    def __init__(self, quantile=0.01, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.quantile = quantile
        self.pattern = None

    def fit(self, documents: list[Document]) -> None:
        X = self.vectorizer.fit_transform([d.page_content for d in documents])
        tfidf_stat_df = pd.DataFrame(
            X.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        ).mean(axis=0)

        words = tfidf_stat_df[tfidf_stat_df < tfidf_stat_df.quantile(self.quantile)]
        less_significant_words = words.index.tolist()

        self.pattern = r"\b({})".format("|".join(less_significant_words))

    def transform(self, text: str) -> str:
        return re.sub(self.pattern, "", text, flags=re.IGNORECASE)

