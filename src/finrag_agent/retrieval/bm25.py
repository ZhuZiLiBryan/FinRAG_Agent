"""
Module for keyword search; BM25 retriever used for keyword search.
"""
import bm25s
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parents[3] / "indexes" / "fiqa_bm25"

class SparseRetriever():
    def __init__(self, index_dir = INDEX_DIR):
        self.index_directory = index_dir

    def search_bm25(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Retrieve top k results from BM25 search as (doc_id, score)
        """
        retriever = bm25s.BM25.load(
            self.index_directory,
            load_corpus=True,
            mmap=True
        )

        # Tokenize query and get top k results
        tokenized_query = bm25s.tokenize(query, stopwords="en")
        docs, scores = retriever.retrieve(tokenized_query, k=k)
        doc_ids = [doc["id"] for doc in docs[0]]

        result = [(doc_ids[i], float(scores[0][i])) for i in range(len(doc_ids))]
        return result
