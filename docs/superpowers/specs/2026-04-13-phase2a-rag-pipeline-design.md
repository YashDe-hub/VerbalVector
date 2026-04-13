# Phase 2A: RAG Pipeline Design

## Goal

Make the note-taking and retrieval system functional: every uploaded recording gets its transcript stored in ChromaDB, and users can query past transcripts via a working `/api/query` endpoint with optional session scoping.

## Architecture

```
/api/upload (existing)
    │
    └── analysis_pipeline ──→ feedback response (existing)
            │
            └── store_transcript() ──→ ChromaDB
                    - chunks transcript into sentences (NLTK)
                    - embeds via SentenceTransformers (all-MiniLM-L6-v2)
                    - stores with metadata: source_id, session_label, timestamp

/api/query (currently a stub)
    │
    ├── source_id filter? ──→ scoped ChromaDB search
    │                              or
    ├── no filter ──→ cross-session ChromaDB search
    │
    └── top-N chunks ──→ RAG prompt ──→ Gemini ──→ answer

/api/sessions (new)
    │
    └── list all stored sessions ──→ source_id, label, timestamp, chunk_count
```

## Components

### 1. Transcript Storage (modify `analysis_pipeline.py`)

The pipeline already calls `store_transcript()` from `manager.py` in a background thread, and it's non-fatal. What needs to change:

- Pass a `source_id` derived from the uploaded filename (the UUID-prefixed safe name that `api.py` generates).
- Accept an optional `session_label` from the upload request and include it in the ChromaDB metadata.
- Ensure `store_transcript()` receives the full transcript text from the STT result. Currently the pipeline has the transcript — verify it's being passed through.

### 2. `/api/query` Endpoint (modify `api.py`)

Replace the current stub with a working implementation. Extend the existing `QueryRequest` Pydantic model (currently only has `query: str`) to include `source_id` and `n_results`.

**Request:**
```json
{
  "query": "What did I say about the project timeline?",
  "source_id": "abc123_presentation.wav",  // optional — scope to one session
  "n_results": 5  // optional, default 5
}
```

**Flow:**
1. Initialize vector store (get or create collection)
2. Search ChromaDB with the query text. If `source_id` is provided, filter by metadata `source == source_id`.
3. If no chunks returned, respond with a message saying no relevant content was found.
4. Construct a RAG prompt: system instruction + retrieved chunks as context + user query.
5. Send to Gemini via the existing `src/services/llm.py` (may need a new function for RAG vs. feedback generation).
6. Return the answer and the retrieved chunk metadata (so the frontend can show sources).

**Response:**
```json
{
  "query": "What did I say about the project timeline?",
  "answer": "In your presentation, you mentioned the timeline is...",
  "sources": [
    {"source_id": "abc123_presentation.wav", "chunk_text": "...", "session_label": "Class presentation"}
  ]
}
```

### 3. `/api/sessions` Endpoint (new in `api.py`)

**Request:** `GET /api/sessions`

**Flow:**
1. Query ChromaDB for all unique `source` values in metadata.
2. For each source, return the label, timestamp, and chunk count.

**Response:**
```json
{
  "sessions": [
    {
      "source_id": "abc123_presentation.wav",
      "session_label": "Class presentation",
      "timestamp": "2026-04-13T14:30:00Z",
      "chunk_count": 42
    }
  ]
}
```

ChromaDB doesn't have a native "list distinct metadata values" operation, so this will likely need a `collection.get()` with a limit, then aggregate unique sources in Python. For the expected scale (tens to low hundreds of sessions), this is fine.

### 4. RAG Prompt (new function in `src/services/llm.py`)

Separate from the existing feedback generation function. The RAG prompt should:

- Instruct Gemini to answer based solely on the provided transcript context
- Include the retrieved chunks with their session labels
- Ask Gemini to cite which session(s) the answer comes from
- If the context doesn't contain the answer, say so rather than hallucinate

### 5. Upload Enhancement (modify `api.py`)

Add an optional `session_label` field to the upload endpoint. This gets passed through the pipeline to `store_transcript()` as metadata. If not provided, auto-generate from the filename and timestamp.

### 6. Search Filtering (modify `manager.py`)

`search_transcripts()` currently takes `query`, `collection`, `n_results`. Add an optional `source_id` parameter. When provided, use ChromaDB's `where` filter: `where={"source": source_id}`.

## Files Changed

| File | Change |
|------|--------|
| `api.py` | Implement `/api/query`, add `/api/sessions`, add `session_label` to upload |
| `src/pipelines/analysis_pipeline.py` | Pass `source_id` and `session_label` to transcript storage |
| `src/vector_store/manager.py` | Add `source_id` filtering to `search_transcripts()`, ensure metadata includes `session_label` |
| `src/services/llm.py` | Add `generate_rag_answer()` function |
| `tests/test_query.py` | Tests for the query endpoint with mocked ChromaDB and Gemini |
| `tests/test_sessions.py` | Tests for the sessions endpoint |

## What Doesn't Change

- The analysis pipeline's core flow (STT → features → emotion → feedback)
- The frontend (all testable via Swagger at `/docs`)
- Config structure (no new env vars needed — Gemini key already exists)
- The upload response format (feedback still returned as before)

## Success Criteria

1. Upload an audio file → transcript appears in ChromaDB with correct metadata
2. `POST /api/query` with a question → returns a relevant answer citing the source session
3. `POST /api/query` with `source_id` → answer scoped to that session only
4. `GET /api/sessions` → lists all stored sessions with metadata
5. Query with no relevant content → returns "no relevant content found" instead of hallucinating
6. ChromaDB failure during upload → pipeline still returns feedback (non-fatal, existing behavior)
