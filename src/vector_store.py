from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions


@dataclass(frozen=True)
class ChromaConfig:
    persist_path: str = "./chroma_db"
    collection_name: str = "candidate_360_kb"
    embedding_model: str = "all-MiniLM-L6-v2"


def build_parallel_lists(all_tagged_chunks: Iterable[dict]) -> tuple[list[str], list[dict], list[str]]:
    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for i, ch in enumerate(all_tagged_chunks):
        docs.append(str(ch["content"]))
        md = {k: v for k, v in ch.items() if k != "content"}
        metadatas.append(md)
        ids.append(f"chunk_{i}")

    return docs, metadatas, ids


def get_collection(config: ChromaConfig) -> Collection:
    client = chromadb.PersistentClient(path=config.persist_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.embedding_model)
    return client.get_or_create_collection(name=config.collection_name, embedding_function=ef)


def reset_collection(config: ChromaConfig) -> Collection:
    client = chromadb.PersistentClient(path=config.persist_path)
    try:
        client.delete_collection(name=config.collection_name)
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.embedding_model)
    return client.get_or_create_collection(name=config.collection_name, embedding_function=ef)


def upsert_chunks(
    collection: Collection,
    *,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    if not (len(documents) == len(metadatas) == len(ids)):
        raise ValueError("documents, metadatas, ids must be parallel lists of equal length")

    # For a fresh index, prefer reset_collection(); otherwise this will add duplicates.
    collection.add(documents=documents, metadatas=metadatas, ids=ids)


def query_top_k(
    collection: Collection,
    query: str,
    *,
    k: int = 10,
) -> tuple[list[str], list[dict]]:
    results: dict[str, Any] = collection.query(query_texts=[query], n_results=k)
    docs = (results.get("documents") or [[]])[0] or []
    mds = (results.get("metadatas") or [[]])[0] or []
    return docs, mds

