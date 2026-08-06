"""Dual-View Retriever: Graph-view and temporal-hierarchy retrieval with evidence closure.

Phase 2 of Zero-Mem integration. All operations zero-LLM-token:
- Graph-view: BM25 + dense matching + PageRank propagation
- Hierarchical-view: coarse-to-fine (local -> episode -> window -> turn)
- Evidence closure: score fusion + neighbor expansion + deduplication
"""

from __future__ import annotations

import math
from collections import Counter
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize

from zero_mem.index_builder import (
    Message,
    Entity,
    TemporalNode,
    ZeroMemIndex,
)


# ---------------------------------------------------------------------------
# BM25 sparse retrieval
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 sparse retrieval over a fixed corpus of texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._vectorizer: TfidfVectorizer | None = None
        self._doc_matrix: np.ndarray | None = None
        self._avgdl: float = 0.0
        self._doc_lens: np.ndarray | None = None

    def fit(self, documents: list[str]) -> "BM25Retriever":
        self._vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        self._doc_matrix = self._vectorizer.fit_transform(documents).toarray()
        self._doc_lens = np.array([len(doc.split()) for doc in documents], dtype=np.float64)
        self._avgdl = self._doc_lens.mean() if len(self._doc_lens) > 0 else 1.0
        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._vectorizer is None or self._doc_matrix is None:
            return []
        q_vec = self._vectorizer.transform([query]).toarray()[0]
        N = self._doc_matrix.shape[0]
        scores = np.zeros(N, dtype=np.float64)
        doc_freq = (self._doc_matrix > 0).sum(axis=0)
        idf = np.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        for term_idx in np.where(q_vec > 0)[0]:
            tf = self._doc_matrix[:, term_idx]
            num = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * self._doc_lens / self._avgdl)
            scores += idf[term_idx] * num / (denom + 1e-9)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Dense embedding retrieval
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense k-NN retrieval using pre-computed embeddings."""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self._nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="cosine")
        self._nn.fit(self.embeddings)

    @classmethod
    def from_index(cls, index: ZeroMemIndex) -> "DenseRetriever | None":
        if index.embeddings is None or index.embeddings.shape[0] == 0:
            return None
        return cls(index.embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        q = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self._nn.kneighbors(q)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            score = float(1.0 - dist)  # cosine similarity
            if score > 0:
                results.append((int(idx), score))
        return results[:top_k]


# ---------------------------------------------------------------------------
# PageRank propagation over entity graph
# ---------------------------------------------------------------------------

def pagerank_propagate(
    graph: nx.Graph,
    seed_entities: list[str],
    alpha: float = 0.85,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Personalized PageRank starting from seed entity nodes.

    Args:
        graph: Entity co-occurrence graph (entity keys as nodes).
        seed_entities: Entity keys to use as personalization seeds.
        alpha: Damping factor (0.85 default from PageRank).
        top_k: Max entities to return.

    Returns:
        Sorted list of (entity_key, PageRank_score).
    """
    if not seed_entities or graph.number_of_nodes() == 0:
        return []

    # Build personalization vector
    personalization = {}
    valid_seeds = [s for s in seed_entities if s in graph]
    if not valid_seeds:
        return []

    weight = 1.0 / len(valid_seeds)
    for seed in valid_seeds:
        personalization[seed] = weight

    try:
        pr = nx.pagerank(graph, alpha=alpha, personalization=personalization,
                         max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        return []

    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [(k, v) for k, v in ranked if k not in set(valid_seeds)][:top_k]


def graph_neighbor_expansion(
    graph: nx.Graph,
    entity_keys: list[str],
    max_hops: int = 2,
) -> list[str]:
    """Expand entity set by graph neighbors up to max_hops."""
    expanded: set[str] = set()
    frontier = set(entity_keys)
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in expanded:
                    next_frontier.add(neighbor)
        expanded.update(next_frontier)
        frontier = next_frontier
    return sorted(expanded)


# ---------------------------------------------------------------------------
# Query parsing (deterministic, zero-LLM)
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    text: str
    entities: list[str]   # entity keys matched via index
    keywords: list[str]   # content keywords for BM25
    intent: str           # "relational", "local", "mixed"


TECH_TERM_ALIASES = {
    "docker": "docker", "json": "json", "api": "api", "ai": "ai",
    "mcp": "mcp", "caprover": "caprover", "github": "github",
    "spacy": "spacy", "python": "python", "kubernetes": "kubernetes",
    "cloudflare": "cloudflare", "bge-m3": "bge-m3", "minilm": "minilm",
}
_QUERY_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how", "does", "did",
    "do", "is", "are", "was", "were", "can", "could", "would", "will",
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "about", "from", "into", "between", "together", "me", "i", "we", "all",
}

def _entity_key_for_text(text: str, index: ZeroMemIndex | None, label: str = "") -> str | None:
    norm = text.lower().strip()
    if norm in TECH_TERM_ALIASES:
        norm = TECH_TERM_ALIASES[norm]
    if index is None:
        return f"{label or 'TECH'}:{norm}"
    candidates = [k for k in index.entity_graph if k.split(":", 1)[-1] == norm]
    return candidates[0] if candidates else None


def parse_query(query: str, index: ZeroMemIndex, spacy_model) -> ParsedQuery:
    """Parse entities, useful lexical terms, and intent deterministically."""
    doc = spacy_model(query[:5000])
    query_entities: list[str] = []
    entity_terms: set[str] = set()
    for ent in doc.ents:
        key = _entity_key_for_text(ent.text, index, ent.label_)
        if key and key not in query_entities:
            query_entities.append(key)
            entity_terms.add(ent.text.lower().strip())

    # Also recognize technical terms that spaCy often labels incorrectly or misses.
    lower_query = query.lower()
    for term in TECH_TERM_ALIASES:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lower_query):
            key = _entity_key_for_text(term, index, "TECH")
            if key and key not in query_entities:
                query_entities.append(key)
            entity_terms.add(term)

    keywords: list[str] = []
    for token in doc:
        term = token.lemma_.lower().strip() or token.text.lower().strip()
        if (token.is_punct or token.like_num or len(term) < 3 or term in _QUERY_STOPWORDS
                or token.is_stop or term in entity_terms or term in TECH_TERM_ALIASES):
            continue
        if token.pos_ in ("NOUN", "PROPN", "ADJ", "VERB") or re.match(r"^[a-z][a-z0-9_-]+$", term):
            if term not in keywords:
                keywords.append(term)
    # Ensure meaningful terms survive models with incomplete POS tagging.
    if not keywords:
        for term in re.findall(r"[a-z][a-z0-9_-]{2,}", lower_query):
            if term not in _QUERY_STOPWORDS and term not in entity_terms and term not in TECH_TERM_ALIASES and term not in keywords:
                keywords.append(term)
    keywords = keywords[:20]

    relational_signals = {"compare", "comparison", "related", "connection", "connect", "link", "relationship", "difference", "across"}
    temporal_signals = {"recent", "recently", "last", "latest", "before", "after", "earlier", "later", "previous", "current", "history", "conversation", "thread"}
    words = set(re.findall(r"[a-z]+", lower_query))
    has_relational_language = bool(words & relational_signals)
    has_temporal_language = bool(words & temporal_signals)
    # Multiple named entities alone is relational; mixed requires an explicit temporal cue too.
    is_comparison = bool(words & {"compare", "comparison", "versus", "vs", "difference"})
    if has_temporal_language and (has_relational_language or len(query_entities) >= 2):
        intent = "mixed"
    elif is_comparison and len(query_entities) >= 2:
        intent = "mixed"
    elif has_relational_language or len(query_entities) >= 2:
        intent = "relational"
    elif has_temporal_language:
        intent = "local"
    else:
        intent = "local" if len(query_entities) <= 1 else "relational"
    return ParsedQuery(query, query_entities, keywords, intent)


# ---------------------------------------------------------------------------
# Coarse-to-fine temporal search
# ---------------------------------------------------------------------------

def temporal_coarse_to_fine(
    query_keywords: list[str],
    parsed_query: ParsedQuery,
    index: ZeroMemIndex,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Coarse-to-fine search through temporal hierarchy.

    local -> episode -> window -> turn, pruning at each level.
    Returns (message_index, relevance_score) pairs.
    """
    tree = index.temporal_tree
    msg_index: dict[str, int] = {m.msg_id: i for i, m in enumerate(index.messages)}

    keyword_set = set(k.lower() for k in query_keywords)
    scores: dict[int, float] = {}

    def score_text(text: str) -> float:
        """Simple keyword-overlap score."""
        text_lower = text.lower()
        hits = sum(1 for kw in keyword_set if kw in text_lower)
        if not hits:
            return 0.0
        # Bonus for density
        return hits / (1 + math.log(len(text_lower.split()) + 1))

    # Level 1: local (thread-level) — pick threads with keyword matches
    local_nodes = tree.get("local", [])
    relevant_locals: list[TemporalNode] = []
    for node in local_nodes:
        s = score_text(node.summary_text)
        if s > 0:
            relevant_locals.append(node)

    if not relevant_locals:
        return []

    # Level 2: episode — within relevant threads
    episode_nodes = tree.get("episode", [])
    episode_map: dict[str, TemporalNode] = {n.node_id: n for n in episode_nodes}
    relevant_episodes: list[TemporalNode] = []
    for local in relevant_locals:
        for child_id in local.children:
            ep = episode_map.get(child_id)
            if ep:
                s = score_text(ep.summary_text)
                if s > 0:
                    relevant_episodes.append(ep)

    if not relevant_episodes:
        return []

    # Level 3: window — within relevant episodes
    window_nodes = tree.get("window", [])
    window_map: dict[str, TemporalNode] = {n.node_id: n for n in window_nodes}
    relevant_windows: list[TemporalNode] = []
    for ep in relevant_episodes:
        for child_id in ep.children:
            win = window_map.get(child_id)
            if win:
                s = score_text(win.summary_text)
                if s > 0:
                    relevant_windows.append(win)

    # Level 4: turn — score individual messages within relevant windows
    turn_nodes = tree.get("turn", [])
    turn_map: dict[str, TemporalNode] = {n.node_id: n for n in turn_nodes}

    for win in relevant_windows:
        for child_id in win.children:
            turn = turn_map.get(child_id)
            if turn and turn.message_ids:
                msg_id = turn.message_ids[0]
                idx = msg_index.get(msg_id)
                if idx is not None:
                    s = score_text(turn.summary_text)
                    if s > 0:
                        scores[idx] = scores.get(idx, 0.0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Graph-view retrieval
# ---------------------------------------------------------------------------

def graph_view_retrieve(
    parsed_query: ParsedQuery,
    index: ZeroMemIndex,
    bm25: BM25Retriever,
    dense: DenseRetriever | None,
    query_embedding: np.ndarray | None = None,
    top_k: int = 10,
    pagerank_alpha: float = 0.85,
) -> list[tuple[int, float]]:
    """Graph-primary retrieval: entity matching -> PageRank propagation ->
    message retrieval via BM25 + dense.

    Returns (message_index, fused_score).
    """
    msg_index: dict[str, int] = {m.msg_id: i for i, m in enumerate(index.messages)}
    scores: dict[int, float] = {}

    # Step 1: PageRank propagate from query entities
    propagated = pagerank_propagate(
        index.entity_graph,
        parsed_query.entities,
        alpha=pagerank_alpha,
        top_k=30,
    )

    # Build expanded entity set
    all_entities = set(parsed_query.entities)
    all_entities.update(k for k, _ in propagated)

    # Step 2: Find messages touching these entities
    relevant_msg_ids: set[str] = set()
    for e in index.entities:
        if e.key in all_entities:
            relevant_msg_ids.add(e.msg_id)

    # Step 3: BM25 over relevant messages
    bm25_messages = [(msg_index[mid], 1.0) for mid in relevant_msg_ids if mid in msg_index]
    bm25_query = " ".join(parsed_query.keywords)
    if bm25_query.strip():
        bm25_messages = bm25.search(bm25_query, top_k=top_k * 3)

    for idx, bm25_score in bm25_messages:
        scores[idx] = scores.get(idx, 0.0) + bm25_score * 0.6

    # Step 4: Dense retrieval (if available)
    if dense is not None and query_embedding is not None:
        dense_results = dense.search(query_embedding, top_k=top_k * 2)
        for idx, dense_score in dense_results:
            scores[idx] = scores.get(idx, 0.0) + dense_score * 0.4

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Evidence closure
# ---------------------------------------------------------------------------

@dataclass
class RetrievedEvidence:
    """Evidence block returned to the reader (LLM only sees this)."""
    message: Message
    score: float
    source_view: str  # "graph", "temporal", or "both"
    related_entities: list[Entity] = field(default_factory=list)
    temporal_context: list[str] = field(default_factory=list)  # surrounding msg ids


def evidence_closure(
    graph_results: list[tuple[int, float]],
    temporal_results: list[tuple[int, float]],
    index: ZeroMemIndex,
    rho: float = 0.5,    # graph weight (0..1); temporal weight = 1 - rho
    top_k: int = 10,
    expand_hops: int = 1,
) -> list[RetrievedEvidence]:
    """Fuse graph and temporal results, expand neighbors, deduplicate.

    rho controls the balance between graph-primary and temporal-primary.
    rho > 0.5: graph-primary (relational queries)
    rho < 0.5: temporal-primary (local queries)
    """

    # Score fusion
    fused: dict[int, dict] = {}
    for idx, score in graph_results:
        fused.setdefault(idx, {"g": 0.0, "t": 0.0})
        fused[idx]["g"] = max(fused[idx]["g"], score)
    for idx, score in temporal_results:
        fused.setdefault(idx, {"g": 0.0, "t": 0.0})
        fused[idx]["t"] = max(fused[idx]["t"], score)

    combined: list[tuple[int, float, str]] = []
    for idx, s in fused.items():
        final = rho * s["g"] + (1 - rho) * s["t"]
        view = "graph" if s["g"] > s["t"] else "temporal"
        if s["g"] > 0 and s["t"] > 0:
            view = "both"
        combined.append((idx, final, view))

    combined.sort(key=lambda x: x[1], reverse=True)
    combined = combined[:top_k]

    # Neighbor expansion (graph walk around matched entities)
    evidence_blocks: list[RetrievedEvidence] = []
    msg_map = {i: m for i, m in enumerate(index.messages)}

    for idx, score, view in combined:
        if idx not in msg_map:
            continue
        msg = msg_map[idx]

        # Find entities in this message
        msg_entities = [e for e in index.entities if e.msg_id == msg.msg_id]

        # Expand entity neighbors
        expanded_keys: set[str] = set()
        for e in msg_entities:
            expanded_keys.update(
                graph_neighbor_expansion(index.entity_graph, [e.key], max_hops=expand_hops)
            )

        # Find temporal context (surrounding messages in same thread)
        temporal_context: list[str] = []
        same_thread = [m for m in index.messages
                       if m.thread_id == msg.thread_id and m.msg_id != msg.msg_id]
        same_thread.sort(key=lambda m: abs(m.turn_index - msg.turn_index))
        temporal_context = [m.msg_id for m in same_thread[:4]]

        evidence_blocks.append(RetrievedEvidence(
            message=msg,
            score=score,
            source_view=view,
            related_entities=[e for e in index.entities if e.key in expanded_keys],
            temporal_context=temporal_context,
        ))

    return evidence_blocks


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class DualViewRetriever:
    """Complete dual-view retrieval engine: parse -> graph-retrieve ->
    temporal-retrieve -> evidence-closure."""

    def __init__(
        self,
        index: ZeroMemIndex,
        spacy_model,
        embedder,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
    ):
        self.index = index
        self.spacy_model = spacy_model
        self.embedder = embedder
        self.bm25 = bm25 or BM25Retriever().fit(
            [m.content for m in index.messages]
        )
        self.dense = dense or DenseRetriever.from_index(index)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rho: float = 0.5,
        graph_only: bool = False,
        temporal_only: bool = False,
    ) -> list[RetrievedEvidence]:
        """Run a full dual-view retrieval.

        Args:
            query: Natural-language query.
            top_k: Max evidence blocks to return.
            rho: Graph-primary weight (0..1). Overridden by query intent.
            graph_only: Use ONLY graph view.
            temporal_only: Use ONLY temporal view.

        Returns:
            Ranked evidence blocks ready for the LLM reader.
        """
        parsed = parse_query(query, self.index, self.spacy_model)

        # Auto-select rho based on query intent
        if parsed.intent == "relational" and not temporal_only:
            rho = 0.7
        elif parsed.intent == "local" and not graph_only:
            rho = 0.3

        # Graph-view retrieval
        graph_results: list[tuple[int, float]] = []
        if not temporal_only:
            q_embed = None
            if self.dense is not None:
                q_embed = self.embedder.encode(
                    [query[:8000]], convert_to_numpy=True, normalize_embeddings=True
                )
                if q_embed is not None:
                    q_embed = q_embed[0]
            graph_results = graph_view_retrieve(
                parsed, self.index, self.bm25, self.dense, q_embed,
                top_k=top_k,
            )

        # Temporal-view retrieval
        temporal_results: list[tuple[int, float]] = []
        if not graph_only:
            temporal_results = temporal_coarse_to_fine(
                parsed.keywords, parsed, self.index, top_k=top_k,
            )

        # Evidence closure
        return evidence_closure(
            graph_results, temporal_results, self.index,
            rho=rho, top_k=top_k,
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def build_retriever(index: ZeroMemIndex) -> DualViewRetriever:
    """Build a DualViewRetriever from an existing ZeroMemIndex."""
    import spacy
    from sentence_transformers import SentenceTransformer

    nlp = spacy.load("en_core_web_sm")
    try:
        embedder = SentenceTransformer(index.embed_model_name)
    except Exception:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return DualViewRetriever(index=index, spacy_model=nlp, embedder=embedder)
