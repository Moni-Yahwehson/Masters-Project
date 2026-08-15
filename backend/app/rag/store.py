"""Local retrieval index over the curated LSD veterinary corpus.

The corpus is small and static, so a TF-IDF + cosine-similarity index is the
right tool: it is deterministic, needs no embedding service or GPU, and is
trivially reproducible for a thesis appendix. If scikit-learn is unavailable the
store falls back to a pure-Python TF-IDF implementation so retrieval — and
therefore grounding — never silently disappears.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..config import get_settings
from ..schemas import DocumentInfo, RetrievedPassage

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.S)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "for",
    "from", "has", "have", "in", "is", "it", "its", "may", "not", "of", "on",
    "or", "that", "the", "their", "there", "these", "they", "this", "to", "was",
    "were", "which", "with", "you", "your",
}

# Farmers and extension workers do not write in textbook vocabulary. Expanding
# the query with the clinical synonyms of everyday terms measurably improves
# ranking on a corpus this small, where a single vocabulary mismatch is enough
# to surface the wrong section.
_QUERY_EXPANSIONS: dict[str, str] = {
    "spread": "transmission transmitted infectious contagious",
    "spreading": "transmission transmitted infectious",
    "catch": "transmission infection contract",
    "caught": "transmission infection",
    "lump": "nodule nodular swelling",
    "lumps": "nodules nodular swellings",
    "bump": "nodule nodular",
    "bumps": "nodules nodular",
    "sore": "lesion ulcer ulcerated",
    "sores": "lesions ulcers ulcerated",
    "scab": "crust necrotic sit-fast",
    "scabs": "crusts necrotic",
    "shot": "vaccination vaccine immunisation",
    "shots": "vaccination vaccine",
    "jab": "vaccination vaccine",
    "injection": "vaccination vaccine",
    "medicine": "treatment supportive therapy antibiotic",
    "cure": "treatment supportive care recovery",
    "treat": "treatment supportive care",
    "prevent": "control vaccination biosecurity prevention",
    "stop": "control prevention biosecurity",
    "test": "laboratory diagnosis PCR confirmation sampling",
    "tests": "laboratory diagnosis PCR confirmation",
    "confirm": "laboratory confirmation PCR diagnosis",
    "report": "notification notifiable authority",
    "flies": "vector arthropod stomoxys mosquito tick",
    "fly": "vector arthropod stomoxys",
    "insect": "vector arthropod mosquito tick",
    "milk": "yield production lactating udder",
    "pregnant": "abortion reproductive breeding",
    "bull": "semen breeding reproductive orchitis",
    "die": "mortality death fatal",
    "died": "mortality death",
    "sick": "clinical signs disease morbidity",
    "fever": "pyrexia temperature febrile",
    "quarantine": "isolation movement control biosecurity",
    # Anatomy and appearance
    "gland": "lymph node lymphadenopathy prescapular enlarged",
    "glands": "lymph nodes lymphadenopathy prescapular enlarged",
    "swollen": "swelling enlargement oedema enlarged",
    "swelling": "oedema enlargement",
    "look": "appearance morphology characteristic",
    "looks": "appearance morphology characteristic",
    "appearance": "morphology characteristic nodules",
    # NB: deliberately no expansion for "skin" — it appears in the disease name
    # itself, so expanding it injects noise into essentially every query.
    # Automated screening vocabulary
    "ai": "CNN classifier automated screening model limitations",
    "app": "automated screening classifier limitations",
    "model": "CNN classifier automated screening limitations",
    "algorithm": "CNN classifier automated screening limitations",
    "photo": "image-based screening photograph classifier limitations",
    "photos": "image-based screening photographs classifier limitations",
    "photograph": "image-based screening classifier limitations",
    "picture": "image-based screening photograph classifier limitations",
    "pictures": "image-based screening photographs classifier limitations",
    "accurate": "accuracy validation calibration performance limitations",
    "accuracy": "validation calibration performance limitations",
    "reliable": "accuracy validation calibration limitations",
    "trust": "accuracy validation limitations confidence",
}


def _expand_query(query: str) -> str:
    """Append clinical synonyms for colloquial terms found in the query."""
    additions: list[str] = []
    for token in _TOKEN_RE.findall(query.lower()):
        expansion = _QUERY_EXPANSIONS.get(token)
        if expansion:
            additions.append(expansion)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


#: Repetition counts for the indexed text, tuned on the query set in
#: `backend/tests/test_retrieval.py`.
#:
#: The *section* heading is the discriminative signal and is weighted heavily.
#: The *document* title is weighted at 1: several titles are long and
#: keyword-dense ("Differential Diagnosis of Nodular Skin Conditions in
#: Cattle"), so up-weighting them made every chunk of that one document match
#: any query mentioning nodules, drowning out the section that actually
#: answered the question.
_SECTION_WEIGHT = 3
_TITLE_WEIGHT = 1


def _indexed_text(title: str, section: str, body: str) -> str:
    """Build the string that actually gets indexed, with headings up-weighted."""
    return f"{title} " * _TITLE_WEIGHT + f"{section} " * _SECTION_WEIGHT + body


#: Ordered (suffix, replacement) pairs — a deliberately small suffix stripper
#: rather than a full Porter stemmer. The corpus is one domain and a few
#: thousand words, and this is enough to collapse the pairs that actually cost
#: us recall: nodule/nodules/nodular, lesion/lesions, morphology/morphological,
#: vaccinate/vaccination. Applied identically to documents and queries, so an
#: over-aggressive stem still matches on both sides.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ological", "olog"),
    ("ologies", "olog"),
    ("ology", "olog"),
    ("ically", "ic"),
    ("ical", "ic"),
    ("ations", "at"),
    ("ation", "at"),
    ("isation", "ise"),
    ("ization", "ise"),
    ("ising", "ise"),
    ("izing", "ise"),
    ("ular", "ul"),
    ("ness", ""),
    ("ies", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

_MIN_STEM_LENGTH = 3


def _stem(token: str) -> str:
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)] + replacement
            if len(stem) >= _MIN_STEM_LENGTH:
                return stem
            return token
    return token


def _tokenize(text: str) -> list[str]:
    return [
        _stem(token)
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _analyzer(text: str) -> list[str]:
    """Unigram + bigram analyzer over stemmed tokens, shared by both backends."""
    tokens = _tokenize(text)
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    section: str
    text: str
    tokens: list[str] = field(default_factory=list)


@dataclass
class Document:
    document_id: str
    title: str
    source: str
    topics: list[str]
    path: Path
    chunks: list[Chunk]


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, raw[match.end():]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body on `## ` headings into (section_title, text) pairs."""
    sections: list[tuple[str, str]] = []
    current_title = "Overview"
    buffer: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if buffer and any(part.strip() for part in buffer):
                sections.append((current_title, "\n".join(buffer).strip()))
            current_title = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer and any(part.strip() for part in buffer):
        sections.append((current_title, "\n".join(buffer).strip()))
    return sections


class KnowledgeStore:
    """TF-IDF retrieval over the markdown knowledge base."""

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.knowledge_dir = Path(knowledge_dir or settings.knowledge_dir)
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._idf: dict[str, float] = {}
        self._chunk_vectors: list[dict[str, float]] = []
        self._vectorizer = None  # sklearn TfidfVectorizer when available
        self._matrix = None
        self._backend = "python-tfidf"
        self._lock = threading.Lock()
        self._loaded = False

    # -- loading ----------------------------------------------------------
    def load(self) -> "KnowledgeStore":
        with self._lock:
            if self._loaded:
                return self
            self.documents = list(self._read_documents())
            self.chunks = [chunk for doc in self.documents for chunk in doc.chunks]
            if self.chunks:
                self._build_index()
            self._loaded = True
            logger.info(
                "Knowledge store loaded: %d documents, %d chunks (%s)",
                len(self.documents),
                len(self.chunks),
                self._backend,
            )
            return self

    def _read_documents(self) -> Iterable[Document]:
        if not self.knowledge_dir.is_dir():
            logger.warning("Knowledge directory not found: %s", self.knowledge_dir)
            return
        for path in sorted(self.knowledge_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(raw)
            document_id = meta.get("id") or path.stem
            title = meta.get("title") or path.stem.replace("-", " ").title()
            source = meta.get("source", "Project knowledge base")
            topics = [
                topic.strip()
                for topic in meta.get("topics", "").strip("[]").split(",")
                if topic.strip()
            ]
            chunks: list[Chunk] = []
            for index, (section, text) in enumerate(_split_sections(body)):
                chunk_text = text.strip()
                if len(chunk_text) < 40:
                    continue
                chunk = Chunk(
                    chunk_id=f"{document_id}#{index:02d}",
                    document_id=document_id,
                    document_title=title,
                    source=source,
                    section=section,
                    text=chunk_text,
                    # Headings are the highest-signal part of a chunk, so repeat
                    # them in the token bag. Without this the body text swamps
                    # the heading and queries land on the wrong section.
                    tokens=_analyzer(_indexed_text(title, section, chunk_text)),
                )
                chunks.append(chunk)
            yield Document(
                document_id=document_id,
                title=title,
                source=source,
                topics=topics,
                path=path,
                chunks=chunks,
            )

    # -- indexing ---------------------------------------------------------
    def _build_index(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Custom analyzer so sklearn and the pure-Python fallback tokenise
            # and stem identically — otherwise retrieval quality would silently
            # change depending on which dependencies happen to be installed.
            self._vectorizer = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
            corpus = [
                _indexed_text(chunk.document_title, chunk.section, chunk.text)
                for chunk in self.chunks
            ]
            self._matrix = self._vectorizer.fit_transform(corpus)
            self._backend = "sklearn-tfidf"
            return
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("scikit-learn unavailable (%s); using built-in TF-IDF", exc)

        total = len(self.chunks)
        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            document_frequency.update(set(chunk.tokens))
        self._idf = {
            term: math.log((1 + total) / (1 + freq)) + 1.0
            for term, freq in document_frequency.items()
        }
        self._chunk_vectors = [self._vectorize(chunk.tokens) for chunk in self.chunks]
        self._backend = "python-tfidf"

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vector = {
            term: (1.0 + math.log(count)) * self._idf.get(term, 1.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    # -- querying ---------------------------------------------------------
    def search(
        self, query: str, top_k: Optional[int] = None, min_score: Optional[float] = None
    ) -> list[RetrievedPassage]:
        self.load()
        if not self.chunks or not query.strip():
            return []

        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        min_score = settings.retrieval_min_score if min_score is None else min_score
        expanded = _expand_query(query)

        if self._vectorizer is not None and self._matrix is not None:
            scores = self._search_sklearn(expanded)
        else:
            scores = self._search_python(expanded)

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        passages: list[RetrievedPassage] = []
        for chunk, score in ranked[:top_k]:
            if score < min_score:
                continue
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    source=chunk.source,
                    section=chunk.section,
                    text=chunk.text,
                    score=round(float(score), 4),
                )
            )
        return passages

    def _search_sklearn(self, query: str) -> list[float]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self._vectorizer.transform([query])
        return cosine_similarity(query_vector, self._matrix)[0].tolist()

    def _search_python(self, query: str) -> list[float]:
        query_vector = self._vectorize(_analyzer(query))
        scores: list[float] = []
        for chunk_vector in self._chunk_vectors:
            if len(query_vector) > len(chunk_vector):
                smaller, larger = chunk_vector, query_vector
            else:
                smaller, larger = query_vector, chunk_vector
            scores.append(
                sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())
            )
        return scores

    # -- introspection ----------------------------------------------------
    def describe(self) -> list[DocumentInfo]:
        self.load()
        return [
            DocumentInfo(
                document_id=doc.document_id,
                title=doc.title,
                source=doc.source,
                chunks=len(doc.chunks),
                topics=doc.topics,
            )
            for doc in self.documents
        ]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        self.load()
        return next(
            (chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None
        )

    @property
    def backend(self) -> str:
        self.load()
        return self._backend

    @property
    def stats(self) -> tuple[int, int]:
        self.load()
        return len(self.documents), len(self.chunks)


_store: Optional[KnowledgeStore] = None


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore().load()
    return _store
