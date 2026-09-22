"""Sparse representations, including train-fitted BM25 weighting with positive IDF."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Provenance and attribution:
# This sparse-matrix BM25 implementation evolved from earlier project notebooks
# and was inspired in part by the computational structure of Sho Iizuka's
# BM25Transformer:
# https://github.com/arosh/BM25Transformer
# Copyright (c) 2018 Sho IIZUKA -- BSD 3-Clause License.
# See THIRD_PARTY_NOTICES.md.
#
# This implementation includes project-specific modifications: collection
# statistics fitted on training data, k1=1.5 and b=0.75 defaults, configurable
# CountVectorizer vocabulary and unigram/bigram features, and use as document
# features for supervised classification.
#
# Its positive IDF,
# log(1 + (N - df + 0.5) / (df + 0.5)),
# is algebraically identical to the form documented by Apache Lucene's
# BM25Similarity. No Lucene source code is incorporated, and Lucene is not
# an executable dependency of this project.
class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, x: sp.spmatrix, y: object = None) -> "BM25Transformer":
        matrix = sp.csr_matrix(x)
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        document_frequency = np.asarray(binary.sum(axis=0)).ravel()
        n_documents = matrix.shape[0]
        self.idf_ = np.log((n_documents - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)
        lengths = np.asarray(matrix.sum(axis=1)).ravel()
        self.avgdl_ = float(lengths.mean()) if n_documents else 0.0
        return self

    def transform(self, x: sp.spmatrix) -> sp.csr_matrix:
        if not hasattr(self, "idf_"):
            raise ValueError("BM25Transformer must be fitted before transform")
        matrix = sp.csr_matrix(x).copy()
        lengths = np.asarray(matrix.sum(axis=1)).ravel()
        repeated_lengths = np.repeat(lengths, np.diff(matrix.indptr))
        average = self.avgdl_ if self.avgdl_ > 0 else 1.0
        data = matrix.data.astype(np.float64, copy=False)
        denominator = data + self.k1 * (1 - self.b + self.b * repeated_lengths / average)
        weighted = data * (self.k1 + 1) / denominator * self.idf_[matrix.indices]
        return sp.csr_matrix((weighted, matrix.indices, matrix.indptr), shape=matrix.shape)


class BM25Vectorizer(BaseEstimator):
    def __init__(
        self,
        max_features: int | None = 100_000,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        ngram_range: tuple[int, int] = (1, 2),
        lowercase: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.k1 = k1
        self.b = b

    def fit_transform(self, texts: list[str]) -> sp.csr_matrix:
        self.count_vectorizer_ = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
        )
        counts = self.count_vectorizer_.fit_transform(texts)
        self.bm25_ = BM25Transformer(self.k1, self.b).fit(counts)
        return self.bm25_.transform(counts)

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        return self.bm25_.transform(self.count_vectorizer_.transform(texts))


def create_sparse_vectorizer(name: str, **kwargs: object):
    common = {
        key: kwargs[key]
        for key in ("max_features", "min_df", "max_df", "ngram_range", "lowercase")
        if key in kwargs
    }
    if "ngram_range" in common:
        common["ngram_range"] = tuple(common["ngram_range"])
    if name == "bow":
        return CountVectorizer(**common)
    if name == "tfidf":
        return TfidfVectorizer(**common)
    if name == "bm25":
        return BM25Vectorizer(**common, k1=float(kwargs.get("k1", 1.5)), b=float(kwargs.get("b", 0.75)))
    raise ValueError(f"Unknown sparse representation: {name}")

