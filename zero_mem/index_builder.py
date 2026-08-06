"""Index Builder: Constructs the dual-view Zero-Mem index from conversation traces.

This module implements Phase 1 of the Zero-Mem integration:
- Thread message extraction and structuring
- spaCy NER pipeline for entity extraction
- Entity-context graph (NetworkX co-occurrence)
- Temporal hierarchy (turn / window / episode / local)
- Dense embedding index (sentence-transformers BGE-M3)

No LLM tokens are consumed anywhere in this module. All operations use
encoder-only compute (spaCy, sentence-transformers) or deterministic
algorithms (NetworkX, scikit-learn).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import spacy
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message from a conversation trace."""
    msg_id: str
    thread_id: str
    role: str           # "user", "assistant", "system", "tool"
    content: str
    timestamp: float    # Unix epoch
    turn_index: int     # position within thread (0-based)
    metadata: dict = field(default_factory=dict)

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class Entity:
    """An entity extracted from messages via NER."""
    text: str
    label: str          # spaCy entity type: PERSON, ORG, GPE, etc.
    msg_id: str
    thread_id: str
    char_start: int
    char_end: int
    turn_index: int

    @property
    def key(self) -> str:
        """Normalized key for deduplication."""
        return f"{self.label}:{self.text.lower().strip()}"


@dataclass
class TemporalNode:
    """A node in the temporal hierarchy."""
    level: str          # "turn", "window", "episode", "local"
    node_id: str
    thread_id: str
    start_turn: int
    end_turn: int
    message_ids: list[str] = field(default_factory=list)
    summary_text: str = ""  # concatenation of messages (not LLM-summarized)
    children: list[str] = field(default_factory=list)


@dataclass
class ZeroMemIndex:
    """The complete dual-view index built from conversation traces."""
    messages: list[Message]
    entities: list[Entity]
    entity_graph: nx.Graph
    temporal_tree: dict[str, list[TemporalNode]]  # level -> nodes
    embeddings: np.ndarray | None
    embed_model_name: str
    built_at: float
    stats: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Serialize the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save messages
        with open(path / "messages.pkl", "wb") as f:
            pickle.dump(self.messages, f)
        # Save entities
        with open(path / "entities.pkl", "wb") as f:
            pickle.dump(self.entities, f)
        # Save graph (gpickle format via pickle for NetworkX 3.x compat)
        with open(path / "entity_graph.pkl", "wb") as f:
            pickle.dump(self.entity_graph, f)
        # Save temporal tree
        with open(path / "temporal_tree.pkl", "wb") as f:
            pickle.dump(self.temporal_tree, f)
        # Save embeddings
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        # Save metadata
        meta = {
            "built_at": self.built_at,
            "embed_model": self.embed_model_name,
            "stats": self.stats,
            "num_messages": len(self.messages),
            "num_entities": len(self.entities),
            "num_graph_nodes": self.entity_graph.number_of_nodes(),
            "num_graph_edges": self.entity_graph.number_of_edges(),
        }
        with open(path / "index_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ZeroMemIndex":
        """Load a serialized index from disk."""
        path = Path(path)
        with open(path / "messages.pkl", "rb") as f:
            messages = pickle.load(f)
        with open(path / "entities.pkl", "rb") as f:
            entities = pickle.load(f)
        with open(path / "entity_graph.pkl", "rb") as f:
            entity_graph = pickle.load(f)
        with open(path / "temporal_tree.pkl", "rb") as f:
            temporal_tree = pickle.load(f)
        embeddings = None
        embed_path = path / "embeddings.npy"
        if embed_path.exists():
            embeddings = np.load(embed_path)
        with open(path / "index_meta.json", "r") as f:
            meta = json.load(f)
        return cls(
            messages=messages,
            entities=entities,
            entity_graph=entity_graph,
            temporal_tree=temporal_tree,
            embeddings=embeddings,
            embed_model_name=meta.get("embed_model", "BAAI/bge-m3"),
            built_at=meta.get("built_at", 0.0),
            stats=meta.get("stats", {}),
        )


# ---------------------------------------------------------------------------
# Index Builder
# ---------------------------------------------------------------------------

class IndexBuilder:
    """Builds a ZeroMemIndex from raw conversation message records.

    Input: list of message dicts with keys: msg_id, thread_id, role,
           content, timestamp. Optional: turn_index, metadata.

    No LLM tokens consumed. All operations are encoder-only or deterministic.
    """

    # spaCy entity labels we care about (others are noise)
    RELEVANT_LABELS = frozenset({
        "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
        "WORK_OF_ART", "FAC", "NORP", "LANGUAGE", "LAW",
    })

    # Temporal hierarchy parameters
    WINDOW_SIZE = 5       # messages per window
    EPISODE_GAP = 600     # seconds between episodes (10 min)

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        embed_model: str = "BAAI/bge-m3",
    ):
        self.nlp = spacy.load(spacy_model, disable=["lemmatizer", "tagger"])
        # Use a smaller model if BGE-M3 is too large; fallback to all-MiniLM
        try:
            self.embedder = SentenceTransformer(embed_model)
            self.embed_model_name = embed_model
        except Exception:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embed_model_name = "all-MiniLM-L6-v2"

    def build(self, raw_messages: list[dict]) -> ZeroMemIndex:
        """Build the complete dual-view index from raw message dicts."""
        t0 = time.time()

        # Step 1: Normalize messages
        messages = self._normalize_messages(raw_messages)

        # Step 2: NER extraction
        entities = self._extract_entities(messages)

        # Step 3: Entity-context graph
        entity_graph = self._build_entity_graph(messages, entities)

        # Step 4: Temporal hierarchy
        temporal_tree = self._build_temporal_hierarchy(messages)

        # Step 5: Dense embedding index
        embeddings = self._build_embeddings(messages)

        elapsed = time.time() - t0
        stats = {
            "build_time_sec": round(elapsed, 3),
            "num_messages": len(messages),
            "num_entities": len(entities),
            "num_unique_entities": len({e.key for e in entities}),
            "num_graph_nodes": entity_graph.number_of_nodes(),
            "num_graph_edges": entity_graph.number_of_edges(),
            "temporal_levels": {lvl: len(nodes) for lvl, nodes in temporal_tree.items()},
            "embedding_dim": embeddings.shape[1] if embeddings is not None else 0,
        }

        return ZeroMemIndex(
            messages=messages,
            entities=entities,
            entity_graph=entity_graph,
            temporal_tree=temporal_tree,
            embeddings=embeddings,
            embed_model_name=self.embed_model_name,
            built_at=time.time(),
            stats=stats,
        )

    # --- Step 1: Message normalization --------------------------------------

    def _normalize_messages(self, raw: list[dict]) -> list[Message]:
        """Convert raw message dicts into structured Message objects."""
        # Group by thread, sort by timestamp, assign turn indices
        by_thread: dict[str, list[dict]] = {}
        for r in raw:
            tid = r.get("thread_id", "default")
            by_thread.setdefault(tid, []).append(r)

        messages: list[Message] = []
        for tid, msgs in by_thread.items():
            msgs.sort(key=lambda m: m.get("timestamp", 0))
            for i, m in enumerate(msgs):
                messages.append(Message(
                    msg_id=m.get("msg_id", f"{tid}_{i}"),
                    thread_id=tid,
                    role=m.get("role", "unknown"),
                    content=m.get("content", ""),
                    timestamp=float(m.get("timestamp", 0)),
                    turn_index=i,
                    metadata=m.get("metadata", {}),
                ))

        # Sort all messages globally by timestamp
        messages.sort(key=lambda m: m.timestamp)
        return messages

    # --- Step 2: NER extraction ---------------------------------------------

    def _extract_entities(self, messages: list[Message]) -> list[Entity]:
        """Run spaCy NER over all messages. Zero LLM tokens consumed."""
        entities: list[Entity] = []
        for msg in messages:
            if not msg.content.strip():
                continue
            doc = self.nlp(msg.content[:100000])  # cap at 100K chars
            for ent in doc.ents:
                if ent.label_ in self.RELEVANT_LABELS:
                    entities.append(Entity(
                        text=ent.text.strip(),
                        label=ent.label_,
                        msg_id=msg.msg_id,
                        thread_id=msg.thread_id,
                        char_start=ent.start_char,
                        char_end=ent.end_char,
                        turn_index=msg.turn_index,
                    ))
        return entities

    # --- Step 3: Entity-context graph ---------------------------------------

    def _build_entity_graph(
        self, messages: list[Message], entities: list[Entity]
    ) -> nx.Graph:
        """Build co-occurrence graph: entities appearing in the same message
        are connected with weighted edges. Entity nodes carry metadata.

        Zero LLM tokens consumed — pure graph construction.
        """
        G = nx.Graph()

        # Group entities by message
        by_msg: dict[str, list[Entity]] = {}
        for e in entities:
            by_msg.setdefault(e.msg_id, []).append(e)

        # Add entity nodes
        seen_keys: set[str] = set()
        for e in entities:
            if e.key not in seen_keys:
                G.add_node(e.key, text=e.text, label=e.label,
                           first_msg=e.msg_id, first_thread=e.thread_id,
                           occurrences=1)
                seen_keys.add(e.key)
            else:
                G.nodes[e.key]["occurrences"] += 1

        # Add co-occurrence edges (entities in the same message)
        for msg_id, msg_entities in by_msg.items():
            unique_keys = sorted({e.key for e in msg_entities})
            for i in range(len(unique_keys)):
                for j in range(i + 1, len(unique_keys)):
                    k1, k2 = unique_keys[i], unique_keys[j]
                    if G.has_edge(k1, k2):
                        G[k1][k2]["weight"] += 1
                        G[k1][k2]["shared_msgs"].append(msg_id)
                    else:
                        G.add_edge(k1, k2, weight=1, shared_msgs=[msg_id])

        return G

    # --- Step 4: Temporal hierarchy -----------------------------------------

    def _build_temporal_hierarchy(
        self, messages: list[Message]
    ) -> dict[str, list[TemporalNode]]:
        """Build the 4-level temporal hierarchy: turn -> window -> episode -> local.

        - turn:   individual messages
        - window: sliding groups of WINDOW_SIZE messages within a thread
        - episode: messages separated by > EPISODE_GAP seconds of silence
        - local:  the entire thread (top of the hierarchy for that thread)

        Zero LLM tokens consumed — pure structural partitioning.
        """
        tree: dict[str, list[TemporalNode]] = {
            "turn": [],
            "window": [],
            "episode": [],
            "local": [],
        }

        # Group by thread
        by_thread: dict[str, list[Message]] = {}
        for m in messages:
            by_thread.setdefault(m.thread_id, []).append(m)

        for tid, msgs in by_thread.items():
            msgs.sort(key=lambda m: m.turn_index)

            # Level: turn (one node per message)
            for m in msgs:
                node_id = f"{tid}:turn:{m.turn_index}"
                tree["turn"].append(TemporalNode(
                    level="turn",
                    node_id=node_id,
                    thread_id=tid,
                    start_turn=m.turn_index,
                    end_turn=m.turn_index,
                    message_ids=[m.msg_id],
                    summary_text=m.content,
                ))

            # Level: window (groups of WINDOW_SIZE)
            for i in range(0, len(msgs), self.WINDOW_SIZE):
                window_msgs = msgs[i:i + self.WINDOW_SIZE]
                node_id = f"{tid}:window:{i // self.WINDOW_SIZE}"
                child_ids = [
                    f"{tid}:turn:{m.turn_index}" for m in window_msgs
                ]
                tree["window"].append(TemporalNode(
                    level="window",
                    node_id=node_id,
                    thread_id=tid,
                    start_turn=window_msgs[0].turn_index,
                    end_turn=window_msgs[-1].turn_index,
                    message_ids=[m.msg_id for m in window_msgs],
                    summary_text=" ".join(m.content for m in window_msgs),
                    children=child_ids,
                ))

            # Level: episode (split by temporal gaps)
            episodes: list[list[Message]] = []
            current_ep: list[Message] = [msgs[0]] if msgs else []
            for i in range(1, len(msgs)):
                gap = msgs[i].timestamp - msgs[i - 1].timestamp
                if gap > self.EPISODE_GAP:
                    episodes.append(current_ep)
                    current_ep = [msgs[i]]
                else:
                    current_ep.append(msgs[i])
            if current_ep:
                episodes.append(current_ep)

            for ep_idx, ep_msgs in enumerate(episodes):
                node_id = f"{tid}:episode:{ep_idx}"
                child_ids = [
                    f"{tid}:window:{m.turn_index // self.WINDOW_SIZE}"
                    for m in ep_msgs
                ]
                # deduplicate children
                child_ids = list(dict.fromkeys(child_ids))
                tree["episode"].append(TemporalNode(
                    level="episode",
                    node_id=node_id,
                    thread_id=tid,
                    start_turn=ep_msgs[0].turn_index,
                    end_turn=ep_msgs[-1].turn_index,
                    message_ids=[m.msg_id for m in ep_msgs],
                    summary_text=" ".join(m.content for m in ep_msgs),
                    children=child_ids,
                ))

            # Level: local (entire thread)
            node_id = f"{tid}:local"
            child_ids = [
                f"{tid}:episode:{i}" for i in range(len(episodes))
            ]
            tree["local"].append(TemporalNode(
                level="local",
                node_id=node_id,
                thread_id=tid,
                start_turn=msgs[0].turn_index if msgs else 0,
                end_turn=msgs[-1].turn_index if msgs else 0,
                message_ids=[m.msg_id for m in msgs],
                summary_text=" ".join(m.content for m in msgs),
                children=child_ids,
            ))

        return tree

    # --- Step 5: Dense embedding index --------------------------------------

    def _build_embeddings(self, messages: list[Message]) -> np.ndarray | None:
        """Build dense embeddings for all messages using sentence-transformers.

        Zero LLM tokens consumed — encoder-only compute.
        """
        texts = [m.content[:8000] for m in messages if m.content.strip()]
        if not texts:
            return None
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_index(
    messages: list[dict],
    save_path: str | Path | None = None,
) -> ZeroMemIndex:
    """Build a ZeroMemIndex from raw message dicts.

    Args:
        messages: List of dicts with keys msg_id, thread_id, role, content,
                  timestamp. Optional: turn_index, metadata.
        save_path: If provided, serialize the index to this directory.

    Returns:
        ZeroMemIndex with entity graph, temporal hierarchy, and embeddings.
    """
    builder = IndexBuilder()
    index = builder.build(messages)
    if save_path:
        index.save(save_path)
    return index
