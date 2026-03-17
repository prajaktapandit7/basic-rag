from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

from src.ingestion import chunk_pages, extract_pdf_pages
from src.vector_store import ChromaConfig, build_parallel_lists, reset_collection, get_collection, upsert_chunks
from src.evaluator import EvalConfig, answer_question


def build_index(*, data_dir: str, chroma: ChromaConfig) -> None:
    pages = extract_pdf_pages(data_dir)
    if not pages:
        raise RuntimeError(f"No PDF text found under: {data_dir}")

    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    documents, metadatas, ids = build_parallel_lists(chunks)

    collection = reset_collection(chroma)
    upsert_chunks(collection, documents=documents, metadatas=metadatas, ids=ids)

    print(f"Indexed {len(documents)} chunks into collection '{chroma.collection_name}'.")


def interactive_qa(*, chroma: ChromaConfig, eval_config: EvalConfig) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

    client = OpenAI(api_key=api_key)
    collection = get_collection(chroma)

    print("Candidate 360 is ready. Type a question, or 'exit' to quit.")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        print()
        print(answer_question(client=client, collection=collection, user_query=q, config=eval_config))


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate 360 Talent Intelligence (Basic RAG)")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDFs (default: data)")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the Chroma index from PDFs")
    parser.add_argument("--chroma-path", default="./chroma_db", help="Chroma persistent path (default: ./chroma_db)")
    parser.add_argument("--collection", default="candidate_360_kb", help="Chroma collection name")
    parser.add_argument("--top-k", type=int, default=10, help="Retriever top-k (default: 10)")
    args = parser.parse_args()

    chroma = ChromaConfig(persist_path=args.chroma_path, collection_name=args.collection)
    eval_config = EvalConfig(top_k=args.top_k)

    if args.rebuild_index:
        build_index(data_dir=args.data_dir, chroma=chroma)

    interactive_qa(chroma=chroma, eval_config=eval_config)


if __name__ == "__main__":
    main()
