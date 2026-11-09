"""
Generate embeddings of documents and store in pgvector.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg
from pgvector.psycopg import register_vector
import pandas as pd
import numpy as np
import torch


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data" / "raw" / "corpus.parquet"

def main() -> None:
    load_dotenv(ROOT_DIR / ".env")

    corpus = pd.read_parquet(DATA_DIR)
    titles = corpus["title"].fillna("").astype(str).to_list()
    texts = corpus["text"].fillna("").astype(str).to_list()
    ids = corpus["_id"].astype(str).to_list()

    # embed concatenation of both title and text to compress collective semantic meaning
    full_text = [f"{title}\n{text}" for title, text in zip(titles, texts)]
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda",
        model_kwargs={"dtype": torch.float16}

    )
    embeddings = embed_documents(documents=full_text, embedding_model=embedding_model) 

    # Store documents/embeddings in sql
    rows = [
        (doc_id, title, content, embedding)
        for doc_id, title, content, embedding
        in zip(ids, titles, texts, embeddings)
    ]

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO documents (id, title, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                rows
            )

    # embed_documents(["Test 1 this is a sentence", "test 2 I love horsies", "oh how the gentle wind beckons through the leaves"])

def connect_db():
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def embed_documents(documents: list[str], embedding_model: SentenceTransformer) -> np.ndarray:
    # Load sentence transformer model; use GPU if exists for speed
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = embedding_model.encode(
        documents,
        batch_size=8,
        show_progress_bar=True,
        truncate_dim=256,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


if __name__ == "__main__":
    main()
