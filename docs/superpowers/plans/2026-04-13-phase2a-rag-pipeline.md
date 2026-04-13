# Phase 2A: RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up transcript storage on every upload and make `/api/query` return real RAG answers from stored transcripts, with optional session scoping.

**Architecture:** Upload → pipeline stores transcript chunks in ChromaDB with metadata → `/api/query` retrieves chunks, builds RAG prompt, sends to Gemini → returns answer with sources. `/api/sessions` lists stored sessions.

**Tech Stack:** FastAPI, ChromaDB, Gemini (google-genai), NLTK, SentenceTransformers

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `config.py` | Modify | Add `CORS_ALLOWED_ORIGINS`, `API_HOST`, `API_PORT`, `API_RELOAD` |
| `api.py` | Modify | Use config values instead of direct `os.getenv()`, implement `/api/query`, add `/api/sessions`, add `session_label` to upload |
| `src/vector_store/manager.py` | Modify | Remove `logging.basicConfig()`, add `source_id` filtering to `search_transcripts()`, add `list_sessions()`, add `session_label` to metadata in `store_transcript()` |
| `src/services/llm.py` | Modify | Add `generate_rag_answer()` function |
| `src/pipelines/analysis_pipeline.py` | Modify | Accept and pass `source_id` and `session_label` to vector storage thread |
| `tests/test_manager.py` | Create | Tests for manager.py filtering and list_sessions |
| `tests/test_rag.py` | Create | Tests for the RAG answer function |
| `tests/test_api_query.py` | Create | Tests for query and sessions endpoints |

---

### Task 0: Fix config centralization violations

**Files:**
- Modify: `config.py`
- Modify: `api.py`
- Modify: `src/vector_store/manager.py`

Three CLAUDE.md violations found during standards review: `api.py` reads `CORS_ALLOWED_ORIGINS`, `API_HOST`, `API_PORT`, `API_RELOAD` via `os.getenv()` instead of `config.py`, and `manager.py` calls `logging.basicConfig()` at module level (should only be called in the application entry point).

- [ ] **Step 1: Add server config values to config.py**

Add to `config.py` after the `MAX_FILE_SIZE_BYTES` line:

```python
# ── Server Settings ─────────────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS: list[str] = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("API_PORT", "5002"))
API_RELOAD: bool = os.environ.get("API_RELOAD", "false").lower() == "true"
```

- [ ] **Step 2: Update api.py to use config values**

Replace `ALLOWED_ORIGINS = os.getenv(...)` (line 20) with:

```python
ALLOWED_ORIGINS = config.CORS_ALLOWED_ORIGINS
```

Replace the `__main__` block (lines 153-157) with:

```python
if __name__ == "__main__":
    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=config.API_RELOAD)
```

Remove the `import os` from api.py since it's still used by `os.path.splitext`, `os.path.basename`, `os.path.join` — keep it.

- [ ] **Step 3: Remove logging.basicConfig() from manager.py**

Delete line 16 in `src/vector_store/manager.py`:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

This line overrides the root logger configuration already set in `api.py`. Each module should only use `logging.getLogger(__name__)`.

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; [ast.parse(open(f).read()) for f in ['config.py', 'api.py', 'src/vector_store/manager.py']]; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add config.py api.py src/vector_store/manager.py
git commit -m "fix(config): centralize CORS, server settings; remove basicConfig from manager

Moves CORS_ALLOWED_ORIGINS, API_HOST, API_PORT, API_RELOAD into config.py
per CLAUDE.md config discipline. Removes logging.basicConfig() from
manager.py — only the application entry point should configure the root logger."
```

---

### Task 1: Add session metadata and filtering to manager.py

**Files:**
- Modify: `src/vector_store/manager.py`
- Create: `tests/test_manager.py`

Currently `store_transcript()` stores metadata with `source` and `chunk_index`. We need to also store `session_label` and `timestamp`. And `search_transcripts()` needs an optional `source_id` filter.

- [ ] **Step 1: Write failing tests for search filtering and list_sessions**

Create `tests/test_manager.py`:

```python
"""Tests for vector store manager — filtering and session listing."""
import time
from unittest.mock import MagicMock, patch


def _make_mock_collection():
    """Create a mock ChromaDB collection."""
    collection = MagicMock()
    collection.name = "transcripts"
    return collection


def test_search_transcripts_with_source_filter():
    """search_transcripts with source_id should pass a where filter."""
    from src.vector_store.manager import search_transcripts

    collection = _make_mock_collection()
    collection.query.return_value = {"documents": [["chunk1", "chunk2"]]}

    results = search_transcripts(
        query="test query",
        collection=collection,
        n_results=3,
        source_id="abc123",
    )

    call_kwargs = collection.query.call_args[1]
    assert call_kwargs["where"] == {"source": "abc123"}
    assert results == ["chunk1", "chunk2"]


def test_search_transcripts_without_source_filter():
    """search_transcripts without source_id should not pass a where filter."""
    from src.vector_store.manager import search_transcripts

    collection = _make_mock_collection()
    collection.query.return_value = {"documents": [["chunk1"]]}

    results = search_transcripts(
        query="test query",
        collection=collection,
        n_results=3,
    )

    call_kwargs = collection.query.call_args[1]
    assert "where" not in call_kwargs
    assert results == ["chunk1"]


def test_store_transcript_includes_session_label():
    """store_transcript should include session_label in metadata."""
    from src.vector_store.manager import store_transcript

    collection = _make_mock_collection()

    store_transcript(
        transcript_text="Hello world. This is a test.",
        source_id="test_source",
        collection=collection,
        session_label="My Presentation",
    )

    call_kwargs = collection.add.call_args[1]
    metadata_list = call_kwargs["metadatas"]
    assert all(m["session_label"] == "My Presentation" for m in metadata_list)
    assert all(m["source"] == "test_source" for m in metadata_list)


def test_list_sessions():
    """list_sessions should return unique sessions with counts."""
    from src.vector_store.manager import list_sessions

    collection = _make_mock_collection()
    ts = time.time()
    collection.get.return_value = {
        "metadatas": [
            {"source": "a", "session_label": "Talk 1", "timestamp": ts},
            {"source": "a", "session_label": "Talk 1", "timestamp": ts},
            {"source": "b", "session_label": "Talk 2", "timestamp": ts + 1},
        ]
    }

    sessions = list_sessions(collection)

    assert len(sessions) == 2
    session_a = next(s for s in sessions if s["source_id"] == "a")
    assert session_a["chunk_count"] == 2
    assert session_a["session_label"] == "Talk 1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_manager.py -v`

Expected: FAIL — `search_transcripts` doesn't accept `source_id`, `store_transcript` doesn't accept `session_label`, `list_sessions` doesn't exist.

- [ ] **Step 3: Add source_id filter to search_transcripts**

In `src/vector_store/manager.py`, update the `search_transcripts` function signature and body:

```python
def search_transcripts(query: str, collection, n_results: int = 5, source_id: str | None = None) -> list[str]:
```

In the `collection.query()` call, conditionally add a `where` filter:

```python
        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents"],
        }
        if source_id:
            query_kwargs["where"] = {"source": source_id}

        results = collection.query(**query_kwargs)
```

- [ ] **Step 4: Add session_label to store_transcript**

Update `store_transcript` signature:

```python
def store_transcript(transcript_text: str, source_id: str, collection, session_label: str = "") -> bool:
```

Update the metadata list creation (around line 139) to include `session_label`:

```python
        metadata = [
            {
                "source": source_id,
                "chunk_index": i,
                "timestamp": time.time(),
                "session_label": session_label,
            }
            for i in range(len(chunks))
        ]
```

- [ ] **Step 5: Add list_sessions function**

Add to `src/vector_store/manager.py`:

```python
def list_sessions(collection) -> list[dict]:
    """Return a list of unique sessions stored in the collection."""
    if not collection:
        return []

    try:
        all_metadata = collection.get(include=["metadatas"])
        metadatas = all_metadata.get("metadatas", [])

        sessions: dict[str, dict] = {}
        for meta in metadatas:
            source = meta.get("source", "")
            if source not in sessions:
                sessions[source] = {
                    "source_id": source,
                    "session_label": meta.get("session_label", ""),
                    "timestamp": meta.get("timestamp", 0),
                    "chunk_count": 0,
                }
            sessions[source]["chunk_count"] += 1

        return sorted(sessions.values(), key=lambda s: s["timestamp"], reverse=True)

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        return []
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_manager.py -v`

Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add src/vector_store/manager.py tests/test_manager.py
git commit -m "feat(vector-store): add source_id filtering, session_label metadata, list_sessions

- search_transcripts() accepts optional source_id for scoped queries
- store_transcript() accepts optional session_label for metadata
- list_sessions() returns unique sessions with chunk counts"
```

---

### Task 2: Add RAG answer generation to llm.py

**Files:**
- Modify: `src/services/llm.py`
- Create: `tests/test_rag.py`

A new function that takes a user query + retrieved context chunks and returns a Gemini-generated answer.

- [ ] **Step 1: Write failing test**

Create `tests/test_rag.py`:

```python
"""Tests for RAG answer generation."""
from unittest.mock import patch, MagicMock


def test_generate_rag_answer_returns_answer():
    """generate_rag_answer should return text from Gemini."""
    mock_response = MagicMock()
    mock_response.text = "Based on your presentation, the timeline was discussed in Q3."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.services.llm.genai") as mock_genai_module:
        mock_genai_module.Client.return_value = mock_client
        from src.services.llm import generate_rag_answer

        result = generate_rag_answer(
            query="What about the timeline?",
            context_chunks=[
                {"text": "We discussed the Q3 timeline.", "source_id": "talk1", "session_label": "Monday standup"},
            ],
        )

    assert result is not None
    assert "answer" in result
    assert "timeline" in result["answer"].lower()


def test_generate_rag_answer_no_context():
    """generate_rag_answer with empty context should return a no-content message."""
    from src.services.llm import generate_rag_answer

    result = generate_rag_answer(
        query="What about the timeline?",
        context_chunks=[],
    )

    assert result is not None
    assert "no relevant" in result["answer"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_rag.py -v`

Expected: FAIL — `generate_rag_answer` doesn't exist.

- [ ] **Step 3: Implement generate_rag_answer**

Add to `src/services/llm.py`:

```python
def generate_rag_answer(
    query: str,
    context_chunks: list[dict],
) -> Optional[dict]:
    """
    Generate an answer to a user query using retrieved transcript context.

    Args:
        query:          The user's question.
        context_chunks: List of dicts with keys: text, source_id, session_label.

    Returns:
        Dict with 'answer' string, or None on error.
    """
    if not context_chunks:
        return {"answer": "No relevant content was found in your stored transcripts."}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai is not installed.")
        return None

    try:
        from config import GEMINI_API_KEY, GEMINI_LLM_MODEL
    except ImportError:
        import os
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
        GEMINI_LLM_MODEL = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.5-flash")

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set.")
        return None

    context_text = "\n\n---\n\n".join(
        f"[Session: {c.get('session_label', 'Unknown')} | Source: {c.get('source_id', 'Unknown')}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a helpful assistant that answers questions based on stored conversation and presentation transcripts.

Answer the user's question using ONLY the context provided below. If the context does not contain enough information to answer, say so clearly — do not guess or make up information.

When citing information, mention which session it came from.

**Context from stored transcripts:**
{context_text}

**User's question:** {query}

**Answer:**"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.3),
        )

        answer_text = response.text.strip() if response.text else None
        if not answer_text:
            logger.error("[RAG] Gemini returned empty response.")
            return None

        return {"answer": answer_text}

    except Exception as e:
        logger.error(f"[RAG] Gemini generation failed: {e}", exc_info=True)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_rag.py -v`

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/services/llm.py tests/test_rag.py
git commit -m "feat(llm): add generate_rag_answer for transcript Q&A

Separate from feedback generation. Takes retrieved context chunks and
a user query, builds a grounded RAG prompt, sends to Gemini with low
temperature. Returns no-content message if context is empty."
```

---

### Task 3: Pass session metadata through the pipeline

**Files:**
- Modify: `src/pipelines/analysis_pipeline.py`

The pipeline already calls `_perform_vector_storage(transcript_text, base_name, collection)`. We need to also pass `session_label` and use the full `source_id` from `api.py` (the UUID-prefixed safe name) instead of just `base_name`.

- [ ] **Step 1: Update _perform_vector_storage to accept session_label**

In `analysis_pipeline.py`, change the function signature at line 134:

```python
def _perform_vector_storage(transcript_text: str, source_id: str, collection, session_label: str = "") -> None:
```

Update the `store_transcript` call at line 138:

```python
        success = store_transcript(transcript_text, source_id, collection, session_label=session_label)
```

- [ ] **Step 2: Update run_analysis_pipeline to accept source_id and session_label**

Change the function signature at line 151:

```python
def run_analysis_pipeline(
    audio_path: str,
    output_dir: str = "analysis_output",
    source_id: str = "",
    session_label: str = "",
) -> Optional[Dict[str, Any]]:
```

Update the vector thread creation (around line 205-209) to use the passed `source_id` (falling back to `base_name`) and pass `session_label`:

```python
    effective_source_id = source_id or base_name

    vector_thread = threading.Thread(
        target=_perform_vector_storage,
        args=(transcript_text, effective_source_id, collection, session_label),
        daemon=True,
    ) if collection else None
```

- [ ] **Step 3: Verify pipeline still works syntactically**

Run: `python -c "import ast; ast.parse(open('src/pipelines/analysis_pipeline.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/pipelines/analysis_pipeline.py
git commit -m "feat(pipeline): accept source_id and session_label for vector storage

Passes through to store_transcript() so transcripts are tagged with
the upload's UUID-prefixed filename and optional session label."
```

---

### Task 4: Implement /api/query, /api/sessions, and upload session_label

**Files:**
- Modify: `api.py`
- Create: `tests/test_api_query.py`

This is the final wiring task — connecting the API endpoints to the manager and LLM functions built in Tasks 1-2.

- [ ] **Step 1: Write failing tests**

Create `tests/test_api_query.py`:

```python
"""Tests for /api/query and /api/sessions endpoints."""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create a test client with mocked config validation."""
    with patch("config.validate"):
        from api import app
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_query_returns_answer(client):
    """POST /api/query should return an answer from RAG."""
    mock_collection = MagicMock()

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.search_transcripts", return_value=["chunk about timelines"]),
        patch("api.generate_rag_answer", return_value={"answer": "The timeline is Q3."}),
    ):
        response = await client.post("/api/query", json={"query": "timeline?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The timeline is Q3."


@pytest.mark.asyncio
async def test_query_with_source_id(client):
    """POST /api/query with source_id should pass it to search."""
    mock_collection = MagicMock()

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.search_transcripts", return_value=["scoped chunk"]) as mock_search,
        patch("api.generate_rag_answer", return_value={"answer": "Scoped answer."}),
    ):
        response = await client.post(
            "/api/query",
            json={"query": "test?", "source_id": "abc123"},
        )

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["source_id"] == "abc123"
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_sessions_returns_list(client):
    """GET /api/sessions should return session list."""
    mock_collection = MagicMock()
    mock_sessions = [
        {"source_id": "a", "session_label": "Talk", "timestamp": 1000, "chunk_count": 5}
    ]

    with (
        patch("api.initialize_vector_store", return_value=mock_collection),
        patch("api.list_sessions", return_value=mock_sessions),
    ):
        response = await client.get("/api/sessions")

    assert response.status_code == 200
    data = response.json()
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["source_id"] == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_api_query.py -v`

Expected: FAIL — endpoints not implemented yet.

- [ ] **Step 3: Update QueryRequest model and add imports**

In `api.py`, update the `QueryRequest` model:

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    source_id: str | None = Field(None, description="Scope query to a specific session")
    n_results: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
```

Add imports at the top of the file (after existing imports):

```python
from src.vector_store.manager import initialize_vector_store, search_transcripts, list_sessions
from src.services.llm import generate_rag_answer
```

- [ ] **Step 4: Implement /api/query**

Replace the stub at line 144-152 with:

```python
@app.post("/api/query")
async def query_transcript(body: QueryRequest):
    collection = initialize_vector_store()
    if not collection:
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    chunks = search_transcripts(
        query=body.query,
        collection=collection,
        n_results=body.n_results,
        source_id=body.source_id,
    )

    if not chunks:
        return {
            "query": body.query,
            "answer": "No relevant content was found in your stored transcripts.",
            "sources": [],
        }

    context_chunks = [{"text": c, "source_id": body.source_id or "all", "session_label": ""} for c in chunks]

    result = await asyncio.to_thread(generate_rag_answer, body.query, context_chunks)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate answer.")

    return {
        "query": body.query,
        "answer": result["answer"],
        "sources": context_chunks,
    }
```

- [ ] **Step 5: Add /api/sessions endpoint**

Add after the query endpoint:

```python
@app.get("/api/sessions")
async def get_sessions():
    collection = initialize_vector_store()
    if not collection:
        raise HTTPException(status_code=503, detail="Vector store unavailable.")

    sessions = list_sessions(collection)
    return {"sessions": sessions}
```

- [ ] **Step 6: Add session_label to upload endpoint**

Update the upload endpoint signature to accept an optional `session_label` form field. In `api.py`, change:

```python
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
```

to:

```python
from fastapi import Form

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_label: str = Form(""),
):
```

Then update the `run_analysis_pipeline` call (around line 96) to pass the metadata:

```python
        analysis_results = await asyncio.to_thread(
            run_analysis_pipeline,
            audio_path=filepath,
            output_dir=ANALYSIS_OUTPUT_FOLDER,
            source_id=safe_name,
            session_label=session_label or f"Upload {safe_name[:8]}",
        )
```

- [ ] **Step 7: Run tests**

Run: `PYTHONPATH=. pytest tests/test_api_query.py -v`

Expected: 3 PASS

- [ ] **Step 8: Run all tests**

Run: `PYTHONPATH=. pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add api.py tests/test_api_query.py
git commit -m "feat(api): implement /api/query RAG endpoint and /api/sessions

- /api/query accepts query, optional source_id, n_results; returns
  Gemini-generated answer from retrieved transcript chunks
- /api/sessions lists stored sessions with metadata
- Upload endpoint accepts optional session_label form field
- Vector store failure returns 503, not 500"
```

---

## Summary

| Task | Files | What |
|------|-------|------|
| 0 | `config.py`, `api.py`, `manager.py` | Fix config centralization violations from standards review |
| 1 | `manager.py`, `tests/test_manager.py` | Search filtering, session metadata, list_sessions |
| 2 | `llm.py`, `tests/test_rag.py` | RAG answer generation function |
| 3 | `analysis_pipeline.py` | Pass source_id + session_label through pipeline |
| 4 | `api.py`, `tests/test_api_query.py` | Wire up endpoints, connect everything |

Task 0 should run first (fixes config before other work). Tasks 1 and 2 are independent and could be implemented in parallel. Task 3 depends on Task 1. Task 4 depends on all.
