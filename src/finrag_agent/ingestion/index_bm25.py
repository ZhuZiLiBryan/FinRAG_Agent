"""
Perform keyword search via BM25, lexical keyword search for sparse retrieval.
"""
import bm25s
from pathlib import Path
import pandas as pd

# Load corpus data
data_dir = Path(__file__).resolve().parents[3] 
corpus = pd.read_parquet(data_dir / "data" / "raw" / "corpus.parquet")

corpus_ids = corpus["_id"].to_list()
corpus_texts = corpus["text"].to_list()

# preserves mapping of original fiqa _id field with the BM25 internal ids
metadata = [{"id": str(doc_id)} for doc_id in corpus_ids]

# tokenize and remove stopwords to reduce noise
tokens = bm25s.tokenize(corpus_texts, stopwords="en")

# build bm25 index
retriever = bm25s.BM25(corpus=metadata)
retriever.index(tokens)

# save index with ID mapping to disk
indexes_dir = data_dir / "indexes" / "fiqa_bm25"
indexes_dir.mkdir(parents=True, exist_ok=True)

retriever.save(data_dir / "indexes" / "fiqa_bm25")
