# conduit-ai

Python SDK for [Conduit](https://github.com/datakailabs/conduit) — the knowledge graph engine.

[![PyPI](https://img.shields.io/pypi/v/conduit-ai?color=blue)](https://pypi.org/project/conduit-ai/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

## Install

```bash
pip install conduit-ai                    # API client only
pip install 'conduit-ai[langchain]'       # + LangChain retriever
pip install 'conduit-ai[local]'           # + embedded engine (DuckDB, no server needed)
pip install 'conduit-ai[all]'             # Everything
```

## Embedded Mode (no server required)

Run a knowledge graph locally — no Docker, no Postgres, no ArangoDB. Just DuckDB under the hood.

```python
from conduit_ai import LocalConduit

conduit = LocalConduit("./my-knowledge-base")
conduit.install_pack("snowflake-2026.04.ckp")

# Graph-augmented search
results = conduit.search("How does Cortex Search work?", limit=5)
for r in results:
    print(f"{r['score']:.3f} [{r['path']}] {r['title']}")

# Formatted context for LLM prompts
context = conduit.context("Delta Live Tables patterns")
```

Topic-scoped installation — only load what you need:

```python
conduit.install_pack("aws-2026.04.ckp", topics=["s3", "iam"])
```

### LangChain Retriever (embedded)

```python
retriever = conduit.as_retriever(limit=8)
docs = retriever.invoke("How do dynamic tables work?")

# Use in any chain
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

## API Client (connect to a Conduit server)

```python
from conduit_ai import ConduitClient

client = ConduitClient(api_key="ck_...", endpoint="http://localhost:4000")

# Ask a question (GraphRAG + LLM synthesis)
answer = client.ask("How does Snowflake Cortex Search work?")
print(answer.answer)
print(f"Sources: {len(answer.sources)}")

# Retrieve context without LLM
ctx = client.context("data pipeline best practices", limit=5)
for result in ctx.results:
    print(f"{result.title} ({result.score:.2f})")
```

### Conversational follow-ups

```python
import uuid

thread_id = str(uuid.uuid4())
answer1 = client.ask("What is Delta Live Tables?", thread_id=thread_id)
answer2 = client.ask("Can I use it with Cortex?", thread_id=thread_id)
# ^ automatically rewritten to: "Can I use Delta Live Tables with Snowflake Cortex?"
```

### Streaming

```python
async for token in client.aask_stream("Compare Databricks and Snowflake for ML"):
    print(token, end="", flush=True)
```

### LangChain Retriever (server-backed)

```python
from conduit_ai.retriever import ConduitRetriever

retriever = ConduitRetriever(
    api_key="ck_...",
    endpoint="http://localhost:4000",
    kai_id="kai_snowflake",   # Optional: scope to a Kai
    limit=8,
)

docs = retriever.invoke("How do I set up change data capture?")
```

## CLI

Installed automatically with `pip install conduit-ai`:

```bash
# Inspect a knowledge pack
conduit inspect snowflake-2026.04.ckp

# Install a pack (full or topic-scoped)
conduit install snowflake-2026.04.ckp
conduit install aws-2026.04.ckp --topics s3,iam,redshift

# Dry run (preview without installing)
conduit install aws-2026.04.ckp --topics s3 --dry-run

# Ask a question
conduit ask "How does Cortex Search work?" --api-key ck_...

# List installed knowledge domains
conduit list
```

## Knowledge Packs

Knowledge packs (`.ckp` files) are portable, versioned units of domain knowledge. Download seed packs from [datakailabs/knowledge-packs](https://github.com/datakailabs/knowledge-packs):

| Pack | Zettels | Description |
|------|---------|-------------|
| `snowflake-2026.04.ckp` | 5,634 | Snowflake platform documentation |
| `aws-2026.04.ckp` | 3,466 | AWS services documentation |
| `databricks-2026.04.ckp` | 4,173 | Databricks platform documentation |
| `genai-2026.04.ckp` | 1,671 | Generative AI patterns and techniques |

## Two Modes

| | Embedded (`LocalConduit`) | Server (`ConduitClient`) |
|---|---|---|
| **Requires** | `pip install 'conduit-ai[local]'` | Running Conduit server |
| **Storage** | DuckDB (single file) | PostgreSQL + ArangoDB |
| **Graph** | In-memory adjacency list | ArangoDB (full AQL) |
| **LLM synthesis** | Bring your own | Built-in |
| **Multi-user** | No | Yes |
| **Scale** | ~50K zettels | ~500K+ |
| **Best for** | Notebooks, prototyping, CLI | Production, teams, chatbots |

## License

Apache-2.0
