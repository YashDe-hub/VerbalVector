# CLAUDE.md

## Config Discipline

Every API key, model name, embedding model, directory path, and port number should go through config.py with an env var fallback. When adding new configurable values, put them in config.py — do not read os.environ directly in service or pipeline code.

## Service Boundaries

- Each external API gets one wrapper module. Swap providers by editing one file — never the pipeline.
- STT and LLM are pipeline-fatal — if either returns None, the pipeline fails and the API returns 500. Do not wrap these in non-fatal handling.
- Optional services (emotion analysis, vector storage) must be non-fatal. Pipeline logs a warning and continues.
- The pipeline calls run_analysis_pipeline synchronously. In async endpoints, always wrap it in asyncio.to_thread() — never call it directly on the event loop.

## Testing

- Mock external services (Deepgram, Gemini, Hume) in tests — never call real APIs.
- Every service can return None or raise. Test both paths.

## Commit Workflow

Commit format: `type(scope): description` (feat|fix|refactor|test|docs|chore)
