"""
Use a cross-encoder from the combined results of bi-encoder based document searching.
Runs a local model because I want to save money, and seeing as I already paid for my GPU I'm going 
to make the most of it :)
"""
from sentence_transformers import CrossEncoder

def rerank_docs(query: str, docs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    pass

def query_docs(doc_ids:list[str]) -> list[tuple[str, str]]:
    """
    Query via document IDs and get the corresponding doc texts.
    """
    pass