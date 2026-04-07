# conduit-ai

Python SDK for [Conduit](https://github.com/datakailabs/conduit) — the knowledge graph engine.

## Install

```bash
pip install conduit-ai                    # Core client
pip install 'conduit-ai[langchain]'       # With LangChain retriever
```

## Quick Start

### Ask a question

```python
from conduit_ai import ConduitClient

client = ConduitClient(api_key="ck_...", endpoint="http://localhost:4000")

answer = client.ask("How does Snowflake Cortex Search work?")
print(answer.answer)
print(f"Sources: {len(answer.sources)}")
```

### Conversational follow-ups

```python
import uuid

thread_id = str(uuid.uuid4())

answer1 = client.ask("What is Delta Live Tables?", thread_id=thread_id)
answer2 = client.ask("Can I use it with Cortex?", thread_id=thread_id)
# ^ automatically rewritten to: "Can I use Delta Live Tables with Snowflake Cortex?"
```

### Retrieve context (no LLM)

```python
ctx = client.context("data pipeline best practices", limit=5)
for result in ctx.results:
    print(f"{result.title} ({result.score:.2f})")
    print(f"  Domains: {result.domains}")
```

### Stream tokens

```python
async for token in client.aask_stream("Compare Databricks and Snowflake for ML"):
    print(token, end="", flush=True)
```

## LangChain Retriever

Drop Conduit into any LangChain/LangGraph chain:

```python
from conduit_ai.retriever import ConduitRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = ConduitRetriever(
    api_key="ck_...",
    endpoint="http://localhost:4000",
    kai_id="kai_snowflake",   # Optional: scope to a Kai
    limit=8,
)

prompt = ChatPromptTemplate.from_template(
    "Answer based on context:\n{context}\n\nQuestion: {question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

result = chain.invoke("How do I set up change data capture?")
```

### Retriever Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | None | Conduit API key |
| `endpoint` | `https://api.conduit.datakai.com` | Conduit server URL |
| `kai_id` | None | Scope retrieval to a specific Kai |
| `limit` | 8 | Max results to retrieve |
| `include_content` | True | Include full zettel content in documents |
| `include_graph_context` | False | Use /context endpoint (includes graph neighbors) instead of /ask |

## Scoped Knowledge (Kais)

Kais are knowledge views — filtered subsets of the graph scoped by domain, topic, or knowledge type.

```python
# Query only Snowflake knowledge
client = ConduitClient(api_key="ck_...", kai_id="kai_snowflake")
answer = client.ask("How do dynamic tables work?")

# Or per-retriever
retriever = ConduitRetriever(api_key="ck_...", kai_id="kai_aws")
```

## Async Support

Every method has an async counterpart:

```python
answer = await client.aask("question")
ctx = await client.acontext("query")

async with ConduitClient(api_key="ck_...") as client:
    answer = await client.aask("question")
```
