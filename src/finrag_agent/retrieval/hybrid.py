"""
Perform both dense and sparse retrieval.
Combine results of dense and sparse retrieval; uses RRF for rank-aware consolidation/ranking.
"""

import collections

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60):
    rrf_scores = collections.defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            rrf = 1.0 / (k + rank)
            rrf_scores[doc_id] += rrf # add bc doc_id can come from multiple lists

    return sorted(rrf_scores.items(), key=lambda x: -x[1])




