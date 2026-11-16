"""
Retrieve dense embeddings from vector db based on similarity
to embedded query.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg
from pgvector.psycopg import register_vector
import numpy as np
import torch


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ROOT_DIR = Path(__file__).resolve().parents[3]

class DenseRetriever:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        load_dotenv(ROOT_DIR / ".env")
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device="cuda",
            model_kwargs={"dtype": torch.float16}
        )
        self.conn = self.connect_db()

    def connect_db(self):
        conn = psycopg.connect(os.environ["DATABASE_URL"])
        register_vector(conn)
        return conn

    def embed_query(self, query: str) -> np.ndarray:
        print(f"Generating embeddings for query...") 
        embeddings = self.embedding_model.encode(
            query,
            prompt_name="query",
            truncate_dim=256,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        return embeddings

    def retrieve(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_embedding = self.embed_query(query)
        print(f"Embedding Size: {query_embedding.shape}")

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    1 - (embedding <=> %s) AS similarity
                FROM documents
                ORDER BY embedding <=> %s 
                LIMIT %s 
                """, 
                (query_embedding, query_embedding, k)
            )

            return cur.fetchall()

    def close_connection(self):
        self.conn.close()



# Testing
# retriever = DenseRetriever()
# print(retriever.retrieve("How should I be investing in domestic stocks?", k=5))
# retriever.close_connection()