"""
Fetch FiQA dataset from HuggingFace BeIR Repos.
Need be run only once.
"""

from pathlib import Path
from datasets import load_dataset

data_directory = Path(__file__).parent / "raw"
data_directory.mkdir(exist_ok=True, parents=True)

corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels = load_dataset("BeIR/fiqa-qrels", split="test")


corpus.to_parquet(data_directory / "corpus.parquet")
queries.to_parquet(data_directory / "queries.parquet")
qrels.to_parquet(data_directory / "qrels.parquet")


if __name__ == "__main__":
    print(f"Corpus:  {len(corpus):>6} docs    -> {data_directory / 'corpus.parquet'}")
    print(f"Queries: {len(queries):>6} queries -> {data_directory / 'queries.parquet'}")
    print(f"Qrels:   {len(qrels):>6} judgments -> {data_directory / 'qrels.parquet'}")