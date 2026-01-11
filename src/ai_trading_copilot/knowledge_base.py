"""Lightweight RAG over a small markdown knowledge base.

Uses TF-IDF + cosine similarity instead of a neural embedding model on
purpose: it needs no model download, runs instantly in CI, and is easy to
reason about - trading definitions and risk-management notes don't need
semantic embeddings to retrieve well. Swapping in a sentence-transformer
or hosted embedding API later is a drop-in change behind `retrieve()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@dataclass
class RetrievedDoc:
    title: str
    path: str
    score: float
    excerpt: str


class KnowledgeBase:
    def __init__(self, directory: Path = DEFAULT_KB_DIR):
        self.directory = Path(directory)
        self._titles: list[str] = []
        self._paths: list[str] = []
        self._texts: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        files = sorted(self.directory.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"No knowledge base documents found in {self.directory}")

        for file in files:
            text = file.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip() if text else file.stem
            self._titles.append(title)
            self._paths.append(str(file.relative_to(self.directory.parents[1])))
            self._texts.append(text)

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def retrieve(self, query: str, k: int = 2) -> list[RetrievedDoc]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = scores.argsort()[::-1][:k]

        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0:
                continue
            text = self._texts[idx]
            excerpt = " ".join(text.split()[:60])
            results.append(
                RetrievedDoc(
                    title=self._titles[idx],
                    path=self._paths[idx],
                    score=round(score, 4),
                    excerpt=excerpt,
                )
            )
        return results
