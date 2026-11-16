"""
Perform both dense and sparse retrieval.
Combine results of dense and sparse retrieval; uses RRF for rank-aware consolidation/ranking.
"""

import collections
from .dense import DenseRetriever
from .bm25 import SparseRetriever

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60):
    rrf_scores = collections.defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf = 1.0 / (k + rank)
            rrf_scores[doc_id] += rrf # add bc doc_id can come from multiple lists

    return sorted(rrf_scores.items(), key=lambda x: -x[1])

def hybrid_search(query:str, candidate_k: int = 50, k: int = 10) -> list[tuple[str, float]]:
    dense_retriever = DenseRetriever()
    sparse_retriever = SparseRetriever()
    dense_docs = [doc_id for doc_id, _ in dense_retriever.retrieve(query, candidate_k)]
    sparse_docs = [doc_id for doc_id, _ in sparse_retriever.retrieve(query, candidate_k)]

    ranked_fusion_res = reciprocal_rank_fusion([dense_docs, sparse_docs])
    return ranked_fusion_res[:k]



# print(hybrid_search("What is the best time to invest in bonds"))



