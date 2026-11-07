"""
Module for keyword search; BM25 retriever used for keyword search.
"""
import bm25s
from pathlib import Path

index_directory = Path(__file__).resolve().parents[3] / "indexes" / "fiqa_bm25"

def search_bm25(query: str, k: int = 10) -> list[tuple[str, float]]:
    """
    Retrieve top k results from BM25 search as (doc_id, score)
    """
    retriever = bm25s.BM25.load(
        index_directory,
        load_corpus=True,
        mmap=True
    )

    # Tokenize query and get top k results
    tokenized_query = bm25s.tokenize(query, stopwords="en")
    docs, scores = retriever.retrieve(tokenized_query, k=k)
    doc_ids = [doc["id"] for doc in docs[0]]

    result = [(doc_ids[i], float(scores[0][i])) for i in range(len(doc_ids))]
    return result
