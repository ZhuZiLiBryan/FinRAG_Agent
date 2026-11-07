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