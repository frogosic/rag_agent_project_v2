# RAG Agent Project v2

A clean, incremental RAG implementation focused on building a deterministic substrate before adding more advanced retrieval and orchestration layers.

The current system supports:

- config-driven ingestion
- document reading
- sectioning
- chunking
- dense embeddings
- Qdrant indexing
- dense retrieval
- optional metadata filtering
- retrieval evaluation
- grounded answer generation
- FastAPI `/ask` endpoint
- persisted run traces for CLI/API calls

---

## Current architecture

```text
raw documents
→ ingestion config
→ Document
→ Section
→ Chunk
→ chunks.jsonl
→ dense embeddings
→ embedded_chunks.jsonl
→ Qdrant collection
→ dense retrieval
→ LLM answer generation
→ persisted ask run JSON
```

High-level components:

```text
config/      YAML runtime configuration
rag/         Python package code
scripts/     executable local scripts
data/        raw corpus, processed files, evals, run traces
```

---

## Repository structure

```text
rag_agent_project_v2/
  config/
    content_types.yaml

  rag/
    api/
      __init__.py
      app.py
      schemas.py

    application/
      __init__.py
      rag_service.py
      run_store.py

    config/
      content_types.py

    domain/
      documents.py

    generation/
      answer_service.py
      llm_client.py

    ingestion/
      ids.py
      readers.py
      sectioners.py
      chunkers.py
      pipeline.py

    indexing/
      embedding_service.py
      qdrant_store.py

  scripts/
    check_config.py
    inspect_ingestion.py
    ingest.py
    embed_chunks.py
    index_qdrant.py
    query_qdrant.py
    eval_retrieval.py
    ask.py

  data/
    raw/
      technical/
      hr_docs/
      support/

    processed/
      chunks.jsonl
      embedded_chunks.jsonl

    eval/
      retrieval_eval.yaml
      results/
        retrieval_eval_dense.json

    runs/
      ask_runs/
```

---

## Design goals

The project is built around a few constraints:

1. Build deterministic pieces first.
2. Keep ingestion, retrieval, generation, and API boundaries separate.
3. Prefer inspectable intermediate artifacts.
4. Avoid early agent orchestration.
5. Avoid silently hiding retrieval failures behind LLM answers.
6. Keep Qdrant, LLMs, evals, and FastAPI loosely coupled.


---

## Environment setup

Create and activate a virtual environment.

Example:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies.

If using `pip`:

```bash
pip install \
  pydantic \
  pydantic-settings \
  pyyaml \
  markdown-it-py \
  sentence-transformers \
  qdrant-client \
  fastapi \
  uvicorn \
  anthropic \
  openai \
  requests \
  python-dotenv
```

---

## Environment variables

Create a local `.env` file.

Anthropic example:

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-7
ANTHROPIC_API_KEY=your_key_here
```

OpenAI example:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.1
OPENAI_API_KEY=your_key_here
```

Ollama example:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

API keys are not passed manually into the app. Provider SDKs read the relevant environment variables.

---

## Qdrant setup
Run:

```bash
docker compose up -d qdrant
```

---

## Content type config

The ingestion config lives here:

```text
config/content_types.yaml
```

Current model:

```text
reader    = how to read file from disk
sectioner = how to split document into logical sections
chunker   = how to split sections into retrieval units
```

Supported reader:

```text
text
```

Supported sectioners:

```text
markdown
plaintext
policy
faq
```

Supported chunkers:

```text
single
paragraph
section_window
```

Logical content types currently include:

```text
technical
hr_policy
hr_faq
support_ticket
support_runbook
```

The system intentionally distinguishes between raw folders and logical document types. For example:

```text
data/raw/hr_docs/medical_leave_faq.txt
```

can be modeled as:

```yaml
content_type: hr_faq
domain: hr
doc_role: faq
```

---

## Domain model

Core domain objects live in:

```text
rag/domain/documents.py
```

The current model is:

```text
Document = file-level text
Section  = logical structure inside a document
Chunk    = final retrieval unit to embed/index
```

Lineage:

```text
Chunk → Section → Document → source file
```

---

## Deterministic IDs

ID helpers live in:

```text
rag/ingestion/ids.py
```

Current strategy:

```text
document ID = content type + source path hash
section ID  = document ID + ordinal + heading slug + short content hash
chunk ID    = section ID + ordinal + short content hash
```

Qdrant point IDs are deterministic UUIDv5 values derived from chunk IDs.

This gives:

```text
same chunks
→ same chunk IDs
→ same Qdrant point IDs
```

This is not full semantic reconciliation. If chunking or source text changes significantly, IDs may change. That is acceptable for the current baseline.

---

## Ingestion

Canonical ingestion command:

```bash
python scripts/ingest.py
```

Output:

```text
data/processed/chunks.jsonl
```

The ingestion flow:

```text
content_types.yaml
→ discover source files
→ reader: file → Document
→ sectioner: Document → Sections
→ chunker: Sections → Chunks
→ write chunks.jsonl
```

---

## Embedding

Embedding service:

```text
rag/indexing/embedding_service.py
```

Current model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Current embedding dimension:

```text
384
```

Default behavior:

```text
normalize_embeddings=True
```

Run:

```bash
python scripts/embed_chunks.py
```

Input:

```text
data/processed/chunks.jsonl
```

Output:

```text
data/processed/embedded_chunks.jsonl
```

Each embedded chunk includes:

```json
{
  "embedding": [...],
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384
}
```

---

## Qdrant indexing

Qdrant store:

```text
rag/indexing/qdrant_store.py
```

Current collection:

```text
rag_chunks
```

Current vector config:

```text
vector name: dense
dimension: 384
distance: cosine
```

Run:

```bash
python scripts/index_qdrant.py
```

This reads:

```text
data/processed/embedded_chunks.jsonl
```

and indexes all chunks into Qdrant.

The current indexing script recreates the collection intentionally. This gives deterministic baseline behavior:

```text
same embedded input
→ recreated collection
→ same point IDs
→ same indexed state
```

Later, this can be split into:

```text
--recreate
--upsert-only
```

---

## Dense retrieval

Basic dense retrieval script:

```bash
python scripts/query_qdrant.py "How do refresh tokens work?"
```

With custom top-k:

```bash
python scripts/query_qdrant.py "Who is eligible for medical leave?" --top-k 3
```

Current retrieval mode:

```text
dense-only vector search
```

Optional metadata filters are supported in package code and API flow.

This is not hybrid retrieval yet.

---

## Retrieval evaluation

Eval cases live here:

```text
data/eval/retrieval_eval.yaml
```

Run:

```bash
python scripts/eval_retrieval.py
```

Output:

```text
data/eval/results/retrieval_eval_dense.json
```

The eval runner supports:

- expected source checks
- expected chunk ID checks
- rank tracking
- rank requirements
- anti-signals
- anti-signal top-k windows
- warning thresholds
- category summaries
- persisted JSON results

Example eval case:

```yaml
- id: eval_002_medical_leave_eligibility
  query: "Who is eligible for medical leave?"
  category: hr_faq

  filters:
    domain: hr
    doc_role: faq

  expected:
    sources_any:
      - medical_leave_faq.txt
    chunk_ids_any:
      - chk_sec_doc_hr_faq_medical_leave_faq_...

  checks:
    require_rank_lte: 1

  anti_signals:
    check_top_k: 3
    sources:
      - parental_leave_policy.txt
      - sabbatical_policy.txt
      - bereavment_leave_policy.txt

  notes: "Should prefer medical leave FAQ over generic leave eligibility sections."
```

Current eval summary example:

```text
total: 6
passed: 6
failed: 0
pass_rate: 100.00%
expected_misses: 0
anti_signal_failures: 0
rank_failures: 0
warnings: 1
```

The warning currently tracks low confidence for exact identifier retrieval, such as `SUP-1108`.

---

## LLM client

LLM client:

```text
rag/generation/llm_client.py
```

Supported providers:

```text
anthropic
openai
ollama
```

Provider selection is controlled by environment variables:

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-7
```

The rest of the application should depend on `LLMClient`, not provider SDKs directly.

---

## Answer generation

Answer service:

```text
rag/generation/answer_service.py
```

Purpose:

```text
query + retrieved chunks
→ grounded LLM answer
```

The answer prompt instructs the model to:

- use only provided context chunks
- avoid inventing facts
- say when context is insufficient
- cite source names when possible

---

## Application service

Main orchestration service:

```text
rag/application/rag_service.py
```

Purpose:

```text
query + top_k + filters
→ embed query
→ retrieve chunks from Qdrant
→ generate answer
→ return structured result
```

Current return shape:

```json
{
  "run_id": "ask_...",
  "query": "...",
  "filters": {},
  "top_k": 5,
  "retrieval_mode": "dense_with_optional_filters",
  "answer": "...",
  "retrieved_chunks": [...]
}
```

This service is used by both:

```text
scripts/ask.py
rag/api/app.py
```

---

## Run persistence

Run store:

```text
rag/application/run_store.py
```

Purpose:

```text
persist ask runs as JSON for debugging, tracing, and future observability
```

Output directory:

```text
data/runs/ask_runs/
```

Each run file contains:

```json
{
  "created_at": "...",
  "run_id": "ask_...",
  "query": "...",
  "filters": {},
  "top_k": 5,
  "retrieval_mode": "dense_with_optional_filters",
  "answer": "...",
  "retrieved_chunks": [...]
}
```

This is intended to support future:

```text
FastAPI trace inspection
LangGraph run/state tracing
agent step logging
tool call auditing
eval comparison
```

Avoid naming this “thought tracking” in code. Prefer:

```text
trace
steps
state
decisions
tool_calls
retrieval_context
```

---

## CLI ask flow

Run:

```bash
python scripts/ask.py "How do refresh tokens work?"
```

With filters:

```bash
python scripts/ask.py "Who is eligible for medical leave?" --domain hr --doc-role faq
```

This writes a run file to:

```text
data/runs/ask_runs/
```

---

## FastAPI app

API app:

```text
rag/api/app.py
```

Schemas:

```text
rag/api/schemas.py
```

Run:

```bash
python -m uvicorn rag.api.app:app --reload
```

Use `python -m uvicorn`, not bare `uvicorn`, to ensure the project virtual environment is used.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

Ask endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do refresh tokens work?",
    "top_k": 5,
    "filters": {}
  }' | jq
```

Filtered example:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is eligible for medical leave?",
    "top_k": 5,
    "filters": {
      "domain": "hr",
      "doc_role": "faq"
    }
  }' | jq
```

The API response includes:

```json
{
  "run_id": "...",
  "query": "...",
  "filters": {},
  "top_k": 5,
  "retrieval_mode": "...",
  "answer": "...",
  "retrieved_chunks": [...]
}
```

The API also persists the run to:

```text
data/runs/ask_runs/
```

---

## Recommended run order

From a clean local setup:

```bash
# 1. Start Qdrant
docker compose up -d qdrant

# 2. Validate config
python scripts/check_config.py

# 3. Ingest raw documents
python scripts/ingest.py

# 4. Embed chunks
python scripts/embed_chunks.py

# 5. Index Qdrant
python scripts/index_qdrant.py

# 6. Run retrieval eval
python scripts/eval_retrieval.py

# 7. Ask through CLI
python scripts/ask.py "How do refresh tokens work?"

# 8. Start API
python -m uvicorn rag.api.app:app --reload
```

---

## Current retrieval mode

The current retrieval mode is:

```text
dense_with_optional_filters
```

Meaning:

```text
query text
→ dense embedding
→ Qdrant vector search
→ optional metadata filters
→ retrieved chunks
```

This is not hybrid retrieval.

Hybrid retrieval would add lexical/sparse retrieval:

```text
dense vector search
+ sparse/BM25-style search
+ score fusion or reranking
```

That is intentionally postponed.

---

## What is intentionally postponed

The project does not currently implement:

```text
Qdrant sparse vectors
BM25/hybrid retrieval
reranking
LangGraph orchestration
multi-agent workflows
semantic reconciliation
overlap chunking
tokenizer-specific token counting
multiple embedding models
multi-collection indexing
auth/authz for API
persistent database-backed run storage
streaming API responses
```

Reason:

```text
Build deterministic ingestion, retrieval, eval, generation, and API substrate first.
Then optimize retrieval.
Then add orchestration.
```

---

## Current known limitations

### 1. Dense retrieval can be noisy

Example:

```text
"Who is eligible for medical leave?"
```

Dense retrieval may also retrieve adjacent leave policies such as parental leave or sabbatical leave.

Metadata filters reduce this:

```json
{
  "domain": "hr",
  "doc_role": "faq"
}
```

### 2. Exact identifiers may be fragile

Queries like:

```text
SUP-1108
POST /auth/refresh
```

can work, but dense retrieval is not ideal for exact identifiers, codes, ticket IDs, endpoint paths, product IDs, or version numbers.

This is a future argument for:

```text
metadata extraction
lexical search
hybrid retrieval
```

### 3. Qdrant persistence depends on Docker storage

If Qdrant is started without a mounted volume, deleting the container deletes the collection.

Use persistent storage:

```yaml
volumes:
  - ./.qdrant/storage:/qdrant/storage
```

