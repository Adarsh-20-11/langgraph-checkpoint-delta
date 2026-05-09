# langgraph-checkpoint-delta

A [LangGraph](https://github.com/langchain-ai/langgraph) checkpointer that persists agent state to **Delta Lake tables** — designed for Databricks and Unity Catalog environments.

```
pip install langgraph-checkpoint-delta[deltalake]   # No Spark required
pip install langgraph-checkpoint-delta[spark]        # For Unity Catalog via Spark
```

---

## Why not Postgres?

The standard answer for LangGraph state persistence is a Postgres checkpointer. In most environments, that's fine. In a Databricks / Lakehouse environment, it creates three compounding problems:

**1. Ephemeral clusters risk state loss**

Databricks Jobs run on clusters that terminate when the job ends — or unexpectedly if the cluster is preempted. An external Postgres connection doesn't survive a cluster restart. Your agent loses its state mid-run.

**2. Infrastructure fragmentation outside Unity Catalog**

Running Postgres means managing VPC peering, connection strings, IAM roles, and credentials — all outside Unity Catalog's governance boundary. Your agent state lives in a silo that your data platform can't see, audit, or govern.

**3. Always-on clusters for in-memory state are expensive**

Keeping a high-RAM cluster alive 24/7 just to hold in-memory agent state is unnecessary cloud spend. Delta Lake is already your storage layer — use it.

**The fix:** persist state directly to Delta Lake tables inside Unity Catalog. Delta's transaction log gives you ACID guarantees. Unity Catalog gives you governance. Delta's Time Travel gives you free debugging. And you can shut your clusters down when they're not working.

Production results from the original implementation:

- **40% TCO reduction** — eliminated always-on high-RAM clusters
- **100% state reliability** — across cluster restarts and job failures  
- **Zero-ops governance** — agent state is just another Unity Catalog table
- **SQL observability** — anyone with SQL access can query agent state in real time
- **Time Travel debugging** — rewind agent memory to any prior version

---

## Quickstart

```python
from langgraph_checkpoint_delta import DeltaCheckpointer

# Auto-detects backend: Spark if a SparkSession is active, else python-deltalake
checkpointer = DeltaCheckpointer(
    table_uri="main.agents.checkpoints",  # Unity Catalog 3-part name (Spark)
    backend="auto",
)

# Wire it into your graph
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [...]}, config)
```

### Local development (no Spark, no Databricks)

```python
checkpointer = DeltaCheckpointer(
    table_uri="/tmp/my_checkpoints",  # Local path
    backend="deltalake",              # Rust-backed, no Spark needed
)
```

### Context manager (mirrors langgraph-checkpoint-postgres API)

```python
# Sync
with DeltaCheckpointer.from_conn_string("main.agents.checkpoints") as cp:
    graph = builder.compile(checkpointer=cp)

# Async
async with DeltaCheckpointer.afrom_conn_string("main.agents.checkpoints") as cp:
    graph = builder.compile(checkpointer=cp)
```

---

## Installation

### Python-deltalake backend (recommended for local dev and non-Databricks)

No Spark required. Uses the Rust-backed `deltalake` library for fast, lightweight reads and writes.

```
pip install langgraph-checkpoint-delta[deltalake]
```

> **Limitation:** Cannot write to Unity Catalog managed tables by name. Use a raw storage path (local, S3, ABFSS, GCS).

### Spark backend (required for Unity Catalog 3-part names)

Requires PySpark and `delta-spark`. Uses an active `SparkSession` for all I/O — which means it works natively on Databricks Jobs without any extra configuration.

```
pip install langgraph-checkpoint-delta[spark]
```

### Both backends

```
pip install langgraph-checkpoint-delta[all]
```

---

## Backend selection

| `backend=` | When to use | Requires |
|---|---|---|
| `"auto"` _(default)_ | Spark if SparkSession active, else python-deltalake | Whichever is installed |
| `"spark"` | Unity Catalog 3-part names, Databricks Jobs | pyspark + delta-spark + active SparkSession |
| `"deltalake"` | Local dev, CI, non-Databricks, raw storage paths | deltalake |

---

## Delta table schema

Checkpoints are stored in a Delta table with the following queryable columns:

| Column | Type | Description |
|---|---|---|
| `thread_id` | string | LangGraph thread identifier |
| `checkpoint_ns` | string | Namespace (supports sub-graphs) |
| `checkpoint_id` | string | Unique checkpoint UUID |
| `parent_checkpoint_id` | string | ID of preceding checkpoint |
| `type` | string | Serializer type identifier |
| `checkpoint` | binary | Serialized agent state blob |
| `metadata` | binary | Serialized LangGraph metadata blob |
| `created_at` | timestamp | UTC creation time |

`thread_id`, `checkpoint_id`, `parent_checkpoint_id`, and `created_at` are stored as typed columns so you can query them directly with SQL — no deserialization required.

---

## SQL observability

Because state lives in a Delta table, anyone with SQL access can inspect it:

```sql
-- See all checkpoints for a thread
SELECT checkpoint_id, parent_checkpoint_id, created_at
FROM main.agents.checkpoints
WHERE thread_id = 'user-123'
ORDER BY created_at DESC;

-- Count active threads in the last hour
SELECT COUNT(DISTINCT thread_id)
FROM main.agents.checkpoints
WHERE created_at > NOW() - INTERVAL 1 HOUR;
```

---

## Time Travel debugging

Delta Lake keeps a full transaction log. You can rewind to any prior version of your agent's state:

```sql
-- See what the checkpoint table looked like at version 42
SELECT * FROM main.agents.checkpoints VERSION AS OF 42
WHERE thread_id = 'user-123';

-- Restore a specific checkpoint from before a bad run
SELECT * FROM main.agents.checkpoints TIMESTAMP AS OF '2026-01-15 14:30:00'
WHERE thread_id = 'user-123';
```

---

## Databricks Jobs vs Model Serving Endpoints

This checkpointer is designed for **Databricks Jobs**, not Model Serving Endpoints.

Model Serving Endpoints have a hard request timeout (~60–120s). Multi-step agentic workflows that call tools, wait on external APIs, or run many LLM turns will exceed this. Databricks Jobs have no such timeout, support async execution natively, and are the right primitive for long-running agent tasks.

---

## FinOps extension (coming in v0.2)

The checkpointer has a schema extension for DBU cost tracking. When enabled, each checkpoint records token counts, model tier, and estimated Databricks Unit cost — enabling exact ROI analysis by joining checkpoint tables with your billing logs.

```python
checkpointer = DeltaCheckpointer(
    table_uri="main.agents.checkpoints",
    include_finops=True,  # Adds dbu_cost_usd, token_input_count, token_output_count, model_tier columns
)
```

---

## Architecture

```
DeltaCheckpointer          # Extends BaseCheckpointSaver — knows nothing about Spark or Arrow
    │
    ├── env_detect.py      # Runtime backend selection ("auto" / "spark" / "deltalake")
    ├── serialization.py   # Wraps LangGraph's own serializer (JsonPlusSerializer)
    ├── schema.py          # Column definitions — drives both backends + FinOps extension
    │
    └── backends/
        ├── base.py              # Abstract interface (sync + async)
        ├── spark_backend.py     # delta-spark: MERGE INTO via SparkSession
        └── deltalake_backend.py # python-deltalake: MERGE via Rust, PyArrow interchange
```

The checkpointer layer is deliberately thin — it only translates between LangGraph's `CheckpointTuple` objects and the flat row dicts that backends understand. All Delta I/O is handled by the backend, which means the checkpointer logic is fully testable without any Delta or Spark dependency installed.

---

## Development

```bash
git clone https://github.com/yourusername/langgraph-checkpoint-delta
cd langgraph-checkpoint-delta

uv venv
source .venv/bin/activate
uv pip install -e ".[dev,deltalake]"

pytest -v
```

Unit tests run against an in-memory `FakeBackend` — no Delta Lake or Spark required.

---

## Compatibility

| Dependency | Version |
|---|---|
| Python | ≥ 3.9 |
| langgraph-checkpoint | ≥ 2.0 |
| deltalake _(optional)_ | ≥ 0.19 |
| pyspark _(optional)_ | ≥ 3.5 |
| delta-spark _(optional)_ | ≥ 3.0 |

---

## License

MIT