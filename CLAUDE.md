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

## Post-Commit Quality Loop

After every meaningful commit (a feature, bugfix, or logical unit of work — not every micro-commit), run this loop before moving on. Skip stages only when the commit makes them impossible (e.g., no diff to simplify on a pure docs commit).

1. **Simplify** — invoke the `simplify` skill on the changed code. Apply suggested fixes that genuinely improve clarity / reduce duplication; reject the rest with reasoning.
2. **Verify** — invoke `superpowers:verification-before-completion`. Run the test suite, type-check, lint. No success claims without evidence.
3. **Code review** — dispatch `pr-review-toolkit:code-reviewer` (and `silent-failure-hunter` if error handling changed, `pr-test-analyzer` if tests changed). Address Critical and Important findings before pushing.
4. **PR** — push the branch and open the PR with a summary, test plan, and rollback notes.
5. **Review PR** — invoke `pr-review-toolkit:review-pr` on the open PR for the multi-agent comprehensive sweep. Apply final fixes if anything new surfaces.

This loop applies inside agentic execution flows (e.g., `subagent-driven-development`) too — the runner agent enforces it at task boundaries, not per-step.
