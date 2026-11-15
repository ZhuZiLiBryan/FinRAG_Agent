# FinRAG_Agent
Production-style financial research assistant built with RAG, hybrid retrieval, reranking, and agentic workflows. Features PostgreSQL/pgvector, FastAPI, LLM evaluation, observability, and benchmarking to optimize retrieval accuracy, latency, and cost.

```
fiqa-ai-engineering/
│
├── data/
│   ├── corpus.parquet
│   ├── queries.parquet
│   └── qrels.parquet
│
├── src/
│   ├── ingestion/
│   │   └── embed.py
|   |   └── index_bm25.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   └── rag.py
│   │
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   └── experiments.py
│   │
│   ├── agents/
│   │   └── research_agent.py
│   │
│   └── api/
│       └── main.py
│
├── frontend/
│   └── ...
│
├── tests/
├── experiments/
├── docker-compose.yml
├── requirements.txt
└── README.md
```


## Set-Up:

1. Download the financial data using:
```
uv run ./data/load_dataset.py
```

2. Download dependencies using:
```
uv sync
```

3. Create your `.env` file.

4. Start docker for Postgres
```
docker compose up -d
```

5. Apply postgres schema:
```
docker compose exec db psql -U finrag -d finrag -f ./sql/schema.sql
```

6. Create inverse index for BM25 using:
```
uv run ./src/finrag_agent/ingestion/index_bm25.py
```

7. Download embedding model and embed documents as pgvector in Postgres db using:
```
uv run ./src/finrag_agent/ingestion/embed.py
```

