"""
Generate embeddings of documents and store in pgvector.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

def main() -> None:
    pass

    # embed_documents(["Test 1 this is a sentence", "test 2 I love horsies", "oh how the gentle wind beckons through the leaves"])


def embed_documents(documents: list[str]) -> None:
    # Load sentence transformer model; use GPU if exists for speed
    print(f"Generating embeddings for {len(documents)} documents...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )


if __name__ == "__main__":
    main()
