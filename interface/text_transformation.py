import re
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseTextTransformation(ABC):
    @abstractmethod
    def transform(self, text: str) -> str:
        pass


class RegexpTextTransformation(BaseTextTransformation):
    def __init__(self, **kwargs):
        self.re_kwargs = kwargs

    def transform(self, text: str) -> str:
        result = text
        for pattern, value in self.re_kwargs:
            result = re.sub(pattern, value, result, flags=re.IGNORECASE)
        return result


class TfidfFilterTextTransformation(BaseTextTransformation):
    def __init__(self, quantile=0.01, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.quantile = quantile
        self.pattern = None
        self.tfidf_stat_df = None

    def fit(self, documents: list[str]) -> None:
        X = self.vectorizer.fit_transform(documents)
        self.tfidf_stat_df = pd.DataFrame(
            X.toarray(), columns=self.vectorizer.get_feature_names_out()
        ).mean(axis=0)

        words = self.tfidf_stat_df[
            self.tfidf_stat_df < self.tfidf_stat_df.quantile(self.quantile)
        ]
        less_significant_words = words.index.tolist()
        self.pattern = r"\b({})".format("|".join(less_significant_words))

    def transform(self, text: str) -> str:
        return re.sub(self.pattern, "", text, flags=re.IGNORECASE).strip()


class StopWordsTextTransformation(BaseTextTransformation):
    def __init__(self, stop_words):
        self.pattern = r"\b({})".format("|".join(stop_words.split(",")))

    def transform(self, text: str) -> str:
        return re.sub(self.pattern, "", text, flags=re.IGNORECASE).strip()
