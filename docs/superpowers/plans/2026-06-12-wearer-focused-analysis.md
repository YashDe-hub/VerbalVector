# Wearer-Focused Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Post-Commit Quality Loop (CLAUDE.md):** after each task's commit, run Simplify → Verify (pytest + vitest + tsc) → code-reviewer (+ silent-failure-hunter for error-handling tasks, pr-test-analyzer for test-heavy tasks). Address Critical/Important before the next task.

**Goal:** Analyze only the *user's* performance in any recording — identified by a one-time voice enrollment — while keeping the full conversation as LLM context.

**Architecture:** New `src/services/speaker_id.py` wraps SpeechBrain ECAPA-TDNN embeddings (local, lazy-loaded). Enrollment endpoints store a single voiceprint under `data/voice_profile/`. The pipeline matches the voiceprint to a diarized speaker after STT, computes features on the user's segments only, and instructs Gemini to coach only that speaker. All of it is **non-fatal**: any speaker-ID failure falls back to today's generic analysis with `speaker_attribution.enabled=false`. Attribution rides inside the transcript JSON, so upload, WS `session_end`, and the poll endpoint inherit it with zero handler changes. Spec: `docs/superpowers/specs/2026-06-11-wearer-focused-analysis-design.md`.

**Tech Stack:** FastAPI + pytest (backend), SpeechBrain (ECAPA-TDNN) + librosa/soundfile + numpy (already partly in stack), React 19 + TypeScript + Vitest (frontend).

---

## File Structure

- `config.py` — 5 new settings (model id, threshold, profile dir, min enroll seconds, model cache dir).
- `requirements.txt` — add `speechbrain`.
- `src/services/speaker_id.py` (new) — single wrapper for the embedding model: profile save/load/delete, `compute_embedding`, `match_user`, `export_segments_wav`.
- `tests/test_speaker_id.py` (new) — service unit tests (model mocked).
- `api.py` — `POST/GET/DELETE /api/enroll`.
- `tests/test_api_enroll.py` (new) — endpoint tests.
- `src/services/llm.py` — `generate_feedback`/`_build_prompt` gain `user_speaker`; wearer-focused prompt section.
- `tests/test_llm_feedback.py` — new prompt tests.
- `src/pipelines/analysis_pipeline.py` — attribution + user-only feature branch.
- `tests/test_pipeline_attribution.py` (new) — pipeline branch tests.
- `frontend/src/api.ts` — `SpeakerAttribution` type + `getEnrollment`/`enrollVoice`/`deleteEnrollment`.
- `frontend/src/components/EnrollmentPanel.tsx` (new) + test.
- `frontend/src/components/VerbalVector.tsx` — render the panel in the input stage.
- `frontend/src/components/ResultsDisplay.tsx` — "You" labels + confidence banners; + tests.

**Backend run commands:** no `python` on PATH — use `venv/bin/python -m pytest`. Frontend: `cd frontend && npx vitest run` / `npx tsc --noEmit`.

---

### Task 1: Config + `speaker_id` service module

**Files:**
- Modify: `config.py` (after `STREAM_AUDIO_DIR`, `config.py:39`)
- Modify: `requirements.txt` (add `speechbrain`)
- Create: `src/services/speaker_id.py`
- Test: `tests/test_speaker_id.py` (new)

- [ ] **Step 1: Add config settings**

In `config.py`, after the `STREAM_AUDIO_DIR` line (`config.py:39`), add:

```python
VOICE_PROFILE_DIR = ROOT_DIR / "data" / "voice_profile"
SPEAKER_MODEL_CACHE_DIR = ROOT_DIR / "data" / "speaker_model"
```

In the model-identifiers section (after `EMBEDDING_MODEL`, `config.py:25`), add:

```python
SPEAKER_EMBED_MODEL: str = os.environ.get("SPEAKER_EMBED_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
SPEAKER_MATCH_THRESHOLD: float = float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.30"))
ENROLL_MIN_SECONDS: float = float(os.environ.get("ENROLL_MIN_SECONDS", "10"))
```

In the "Ensure runtime directories exist" block (`config.py:42-45`), add:

```python
VOICE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
```

(Do NOT mkdir the model cache dir — SpeechBrain creates it on first download.)

Append `speechbrain` on its own line to `requirements.txt`.

- [ ] **Step 2: Write failing service tests**

Create `tests/test_speaker_id.py`:

```python
"""Tests for the speaker_id service. The SpeechBrain model is always mocked."""
import json
import numpy as np
import pytest
from unittest.mock import patch

from src.services import speaker_id


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    """Point the profile store at a temp dir."""
    monkeypatch.setattr(speaker_id, "_profile_dir", lambda: tmp_path)
    return tmp_path


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------- profile store ----------------

def test_save_and_load_profile_roundtrip(profile_dir):
    emb = _norm([1.0, 2.0, 3.0])
    meta = speaker_id.save_profile(emb, duration_seconds=31.5)
    assert meta["duration_seconds"] == 31.5
    assert (profile_dir / "embedding.npy").exists()
    assert (profile_dir / "profile.json").exists()
    loaded = speaker_id.load_profile()
    assert np.allclose(loaded, emb)


def test_load_profile_returns_none_when_absent(profile_dir):
    assert speaker_id.load_profile() is None
    assert speaker_id.get_profile_meta() is None


def test_delete_profile(profile_dir):
    speaker_id.save_profile(_norm([1.0, 0.0]), duration_seconds=12.0)
    assert speaker_id.delete_profile() is True
    assert speaker_id.load_profile() is None
    assert speaker_id.delete_profile() is False  # already gone


def test_load_profile_corrupt_returns_none(profile_dir):
    (profile_dir / "embedding.npy").write_text("not a numpy file")
    assert speaker_id.load_profile() is None  # never raises


# ---------------- matching ----------------

def _utt(speaker, start, end, text="hi"):
    return {"speaker": speaker, "text": text, "start": start, "end": end, "confidence": 0.9}


def test_match_user_picks_highest_similarity():
    profile = _norm([1.0, 0.0, 0.0])
    fake_embeddings = {0: _norm([0.0, 1.0, 0.0]), 1: _norm([0.9, 0.1, 0.0])}
    utts = [_utt(0, 0.0, 5.0), _utt(1, 5.0, 9.0)]
    with patch.object(speaker_id, "_embed_speaker", side_effect=lambda a, s, spk: fake_embeddings[spk]):
        result = speaker_id.match_user("x.wav", utts, profile)
    assert result["user_speaker"] == 1
    assert result["confidence"] > 0.9
    assert result["low_confidence"] is False


def test_match_user_low_confidence_below_threshold():
    profile = _norm([1.0, 0.0, 0.0])
    # best similarity ~0.1 < threshold 0.30
    with patch.object(speaker_id, "_embed_speaker", return_value=_norm([0.1, 1.0, 0.0])):
        result = speaker_id.match_user("x.wav", [_utt(0, 0.0, 5.0)], profile)
    assert result["user_speaker"] == 0
    assert result["low_confidence"] is True


def test_match_user_returns_none_when_no_embeddings():
    with patch.object(speaker_id, "_embed_speaker", return_value=None):
        assert speaker_id.match_user("x.wav", [_utt(0, 0.0, 5.0)], _norm([1.0, 0.0])) is None


def test_match_user_returns_none_on_exception():
    with patch.object(speaker_id, "_embed_speaker", side_effect=RuntimeError("boom")):
        assert speaker_id.match_user("x.wav", [_utt(0, 0.0, 5.0)], _norm([1.0, 0.0])) is None


def test_match_user_caps_segment_seconds():
    """Segments passed to embedding are capped at ~30s per speaker."""
    captured = {}

    def fake_embed(audio_path, segments, spk):
        captured[spk] = segments
        return _norm([1.0, 0.0])

    utts = [_utt(0, float(i * 10), float(i * 10 + 10)) for i in range(10)]  # 100s total
    with patch.object(speaker_id, "_embed_speaker", side_effect=fake_embed):
        speaker_id.match_user("x.wav", utts, _norm([1.0, 0.0]))
    total = sum(e - s for s, e in captured[0])
    assert total <= speaker_id.MAX_MATCH_SECONDS + 0.01
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_speaker_id.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.speaker_id'`.

- [ ] **Step 4: Implement the service**

Create `src/services/speaker_id.py`:

```python
"""
Speaker identification service (single wrapper for the embedding model).

Wraps SpeechBrain ECAPA-TDNN speaker embeddings. Used for:
  - one-time voice enrollment (profile stored on disk)
  - matching the enrolled user against diarized speakers per recording

This service is OPTIONAL and NON-FATAL: every public function returns None
on failure; callers fall back to generic whole-recording analysis.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MAX_MATCH_SECONDS = 30.0   # per-speaker audio cap when matching
_TARGET_SR = 16000
_MIN_EMBED_SECONDS = 0.5   # refuse to embed less than this much audio

_classifier = None  # lazy singleton


# ---------------------------------------------------------------------------
# Profile store (single profile — personal tool)
# ---------------------------------------------------------------------------

def _profile_dir() -> Path:
    from config import VOICE_PROFILE_DIR
    return Path(VOICE_PROFILE_DIR)


def save_profile(embedding: np.ndarray, duration_seconds: float) -> Dict[str, Any]:
    """Persist the voiceprint. Overwrites any existing profile (re-enrollment)."""
    from config import SPEAKER_EMBED_MODEL
    d = _profile_dir()
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embedding.npy", embedding.astype(np.float32))
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "model": SPEAKER_EMBED_MODEL,
    }
    (d / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("[SpeakerID] Profile saved (%.1fs of audio).", duration_seconds)
    return meta


def load_profile() -> Optional[np.ndarray]:
    """The enrolled voiceprint, or None if absent/unreadable (never raises)."""
    path = _profile_dir() / "embedding.npy"
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception as e:
        logger.warning("[SpeakerID] Could not load profile embedding: %s", e)
        return None


def get_profile_meta() -> Optional[Dict[str, Any]]:
    path = _profile_dir() / "profile.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[SpeakerID] Could not read profile meta: %s", e)
        return None


def delete_profile() -> bool:
    """Remove the profile. True if something was deleted."""
    d = _profile_dir()
    deleted = False
    for name in ("embedding.npy", "profile.json"):
        p = d / name
        if p.exists():
            p.unlink()
            deleted = True
    return deleted


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _get_classifier():
    """Lazy-load the SpeechBrain encoder (first call downloads ~80MB)."""
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from config import SPEAKER_EMBED_MODEL, SPEAKER_MODEL_CACHE_DIR
        logger.info("[SpeakerID] Loading embedding model %s ...", SPEAKER_EMBED_MODEL)
        _classifier = EncoderClassifier.from_hparams(
            source=SPEAKER_EMBED_MODEL,
            savedir=str(SPEAKER_MODEL_CACHE_DIR),
        )
    return _classifier


def _load_audio(audio_path: str, segments: Optional[List[Tuple[float, float]]] = None) -> Optional[np.ndarray]:
    """Load mono 16k audio; optionally only the given (start, end) slices, concatenated."""
    import librosa
    wav, _sr = librosa.load(audio_path, sr=_TARGET_SR, mono=True)
    if segments:
        parts = []
        for start, end in segments:
            s, e = int(start * _TARGET_SR), int(end * _TARGET_SR)
            if e > s:
                parts.append(wav[s:e])
        if not parts:
            return None
        wav = np.concatenate(parts)
    return wav


def compute_embedding(
    audio_path: str,
    segments: Optional[List[Tuple[float, float]]] = None,
) -> Optional[np.ndarray]:
    """L2-normalised speaker embedding of the audio (or just the given segments).
    Returns None on any failure or if there is too little audio."""
    try:
        wav = _load_audio(audio_path, segments)
        if wav is None or wav.size < int(_TARGET_SR * _MIN_EMBED_SECONDS):
            logger.warning("[SpeakerID] Too little audio to embed (%s).", audio_path)
            return None
        import torch
        clf = _get_classifier()
        emb = clf.encode_batch(torch.from_numpy(wav).unsqueeze(0))
        emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm
    except Exception as e:
        logger.warning("[SpeakerID] Embedding failed for %s: %s", audio_path, e)
        return None


def _embed_speaker(audio_path: str, segments: List[Tuple[float, float]], speaker: int) -> Optional[np.ndarray]:
    """Indirection point so tests can patch per-speaker embedding."""
    return compute_embedding(audio_path, segments)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_user(
    audio_path: str,
    utterances: List[Dict[str, Any]],
    profile_embedding: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Find which diarized speaker is the enrolled user.

    Always returns the best-scoring speaker; low_confidence=True when the best
    cosine similarity is below SPEAKER_MATCH_THRESHOLD. None on failure.
    """
    try:
        from config import SPEAKER_MATCH_THRESHOLD

        # Gather up to MAX_MATCH_SECONDS of segments per speaker
        per_speaker: Dict[int, List[Tuple[float, float]]] = {}
        per_speaker_seconds: Dict[int, float] = {}
        for u in utterances:
            spk = u.get("speaker")
            if spk is None:
                continue
            used = per_speaker_seconds.get(spk, 0.0)
            if used >= MAX_MATCH_SECONDS:
                continue
            start, end = float(u["start"]), float(u["end"])
            length = min(end - start, MAX_MATCH_SECONDS - used)
            if length <= 0:
                continue
            per_speaker.setdefault(spk, []).append((start, start + length))
            per_speaker_seconds[spk] = used + length

        if not per_speaker:
            return None

        best_speaker, best_sim = None, -2.0
        for spk, segs in per_speaker.items():
            emb = _embed_speaker(audio_path, segs, spk)
            if emb is None:
                continue
            sim = float(np.dot(emb, profile_embedding))
            if sim > best_sim:
                best_speaker, best_sim = spk, sim

        if best_speaker is None:
            return None

        return {
            "user_speaker": best_speaker,
            "confidence": round(best_sim, 3),
            "low_confidence": best_sim < SPEAKER_MATCH_THRESHOLD,
        }
    except Exception as e:
        logger.warning("[SpeakerID] match_user failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Audio export (user-only WAV for feature extraction)
# ---------------------------------------------------------------------------

def export_segments_wav(
    audio_path: str,
    segments: List[Tuple[float, float]],
    out_path: str,
) -> Optional[str]:
    """Concatenate the given segments into a 16k mono WAV. None on failure."""
    try:
        import soundfile as sf
        wav = _load_audio(audio_path, segments)
        if wav is None or wav.size == 0:
            return None
        sf.write(out_path, wav, _TARGET_SR)
        return out_path
    except Exception as e:
        logger.warning("[SpeakerID] export_segments_wav failed: %s", e)
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_speaker_id.py -v`
Expected: PASS (9 passed). Note: tests never load SpeechBrain (everything patches `_embed_speaker` or uses the profile store).

- [ ] **Step 6: Run the full backend suite + install dep**

Run: `venv/bin/pip install speechbrain` then `venv/bin/python -m pytest tests/ -q`
Expected: all pass (no regressions; config import additions are inert).

- [ ] **Step 7: Commit**

```bash
git add config.py requirements.txt src/services/speaker_id.py tests/test_speaker_id.py
git commit -m "feat(speaker-id): add voiceprint service (ECAPA embeddings, profile store, matching)"
```

---

### Task 2: Enrollment API — `POST/GET/DELETE /api/enroll`

**Files:**
- Modify: `api.py` (import near `from src.services.audio_assembler import AudioAssembler`; routes after `get_session_result`)
- Test: `tests/test_api_enroll.py` (new)

- [ ] **Step 1: Write failing endpoint tests**

Create `tests/test_api_enroll.py`:

```python
"""Tests for the /api/enroll endpoints. speaker_id internals are mocked."""
import io
import numpy as np
import pytest
import pytest_asyncio
from unittest.mock import patch


@pytest_asyncio.fixture
async def client():
    with patch("config.validate"):
        import api
        from httpx import AsyncClient, ASGITransport
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _wav_upload(name="enroll.wav"):
    return {"file": (name, io.BytesIO(b"RIFF....WAVEfmt fake"), "audio/wav")}


@pytest.mark.asyncio
async def test_enroll_happy_path(client):
    meta = {"created_at": "2026-06-12T00:00:00+00:00", "duration_seconds": 31.0, "model": "m"}
    with (
        patch("api.speaker_id.compute_embedding", return_value=np.ones(3, dtype=np.float32)),
        patch("api.speaker_id.save_profile", return_value=meta) as save,
        patch("api.librosa.get_duration", return_value=31.0),
    ):
        resp = await client.post("/api/enroll", files=_wav_upload())
    assert resp.status_code == 200
    assert resp.json()["duration_seconds"] == 31.0
    save.assert_called_once()


@pytest.mark.asyncio
async def test_enroll_too_short_returns_400(client):
    with patch("api.librosa.get_duration", return_value=3.0):
        resp = await client.post("/api/enroll", files=_wav_upload())
    assert resp.status_code == 400
    assert "least" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_enroll_bad_extension_returns_400(client):
    resp = await client.post(
        "/api/enroll", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_enroll_embedding_failure_returns_500(client):
    with (
        patch("api.librosa.get_duration", return_value=31.0),
        patch("api.speaker_id.compute_embedding", return_value=None),
    ):
        resp = await client.post("/api/enroll", files=_wav_upload())
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_get_enroll_status(client):
    meta = {"created_at": "x", "duration_seconds": 31.0, "model": "m"}
    with patch("api.speaker_id.get_profile_meta", return_value=meta):
        resp = await client.get("/api/enroll")
    assert resp.status_code == 200
    assert resp.json() == meta


@pytest.mark.asyncio
async def test_get_enroll_404_when_absent(client):
    with patch("api.speaker_id.get_profile_meta", return_value=None):
        resp = await client.get("/api/enroll")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_enroll(client):
    with patch("api.speaker_id.delete_profile", return_value=True):
        resp = await client.delete("/api/enroll")
    assert resp.status_code == 204
    with patch("api.speaker_id.delete_profile", return_value=False):
        resp = await client.delete("/api/enroll")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_api_enroll.py -v`
Expected: FAIL — 404/405 (routes absent) and `AttributeError` for `api.speaker_id` / `api.librosa`.

- [ ] **Step 3: Implement the endpoints**

In `api.py`, extend the service imports (next to `from src.services.audio_assembler import AudioAssembler`):

```python
from src.services import speaker_id
import librosa
```

Add the routes immediately after the `get_session_result` function:

```python
@app.post("/api/enroll")
async def enroll_voice(file: UploadFile = File(...)):
    """One-time voice enrollment: store the user's voiceprint (single profile)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")

    safe_name = f"enroll_{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    total_bytes = 0
    try:
        with open(filepath, "wb") as f:
            while chunk := await file.read(8192):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="File too large.")
                f.write(chunk)

        duration = await asyncio.to_thread(librosa.get_duration, path=filepath)
        if duration < config.ENROLL_MIN_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Enrollment audio must be at least {config.ENROLL_MIN_SECONDS:.0f} seconds "
                       f"(got {duration:.1f}s). Record ~30 seconds of normal speech.",
            )

        embedding = await asyncio.to_thread(speaker_id.compute_embedding, filepath)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Could not process enrollment audio.")

        meta = speaker_id.save_profile(embedding, duration_seconds=round(duration, 1))
        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Enrollment failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Enrollment failed.")
    finally:
        try:
            os.remove(filepath)  # enrollment audio is transient — only the embedding persists
        except OSError:
            pass


@app.get("/api/enroll")
async def get_enrollment():
    meta = speaker_id.get_profile_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail="No voice profile enrolled.")
    return meta


@app.delete("/api/enroll", status_code=204)
async def delete_enrollment():
    if not speaker_id.delete_profile():
        raise HTTPException(status_code=404, detail="No voice profile enrolled.")
    return None
```

(Blocking work — `librosa.get_duration`, `compute_embedding` — is wrapped in `asyncio.to_thread` per CLAUDE.md's event-loop rule.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_api_enroll.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Full backend suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api.py tests/test_api_enroll.py
git commit -m "feat(api): add voice enrollment endpoints (POST/GET/DELETE /api/enroll)"
```

---

### Task 3: Wearer-focused LLM prompt

**Files:**
- Modify: `src/services/llm.py` (`generate_feedback` signature `llm.py:31-37`, `_build_prompt` `llm.py:193-295`)
- Test: `tests/test_llm_feedback.py` (append)

- [ ] **Step 1: Write failing prompt tests**

Append to `tests/test_llm_feedback.py` (mirror its existing import style):

```python
def test_build_prompt_wearer_mode_coaches_only_user_speaker():
    from src.services.llm import _build_prompt
    utterances = [
        {"speaker": 0, "text": "How was the demo?", "start": 0.0, "end": 1.5, "confidence": 0.9},
        {"speaker": 1, "text": "It went well, we shipped it.", "start": 1.6, "end": 3.9, "confidence": 0.9},
    ]
    prompt = _build_prompt(
        transcript="How was the demo? It went well, we shipped it.",
        features={"words_per_minute": 140.0},
        emotion_scores=None,
        utterances=utterances,
        user_speaker=1,
    )
    # Coach-only instruction present and targeted
    assert "Speaker 1" in prompt
    assert "ONLY Speaker 1" in prompt or "only Speaker 1" in prompt
    # Full conversation kept as context (other speaker's words included)
    assert "How was the demo?" in prompt
    # Features clarified as user-only
    assert "only from Speaker 1" in prompt
    # Generic multi-speaker coaching instruction must NOT appear in wearer mode
    assert "address each speaker by their ID" not in prompt


def test_build_prompt_without_user_speaker_keeps_generic_behavior():
    from src.services.llm import _build_prompt
    utterances = [
        {"speaker": 0, "text": "A.", "start": 0.0, "end": 1.0, "confidence": 0.9},
        {"speaker": 1, "text": "B.", "start": 1.1, "end": 2.0, "confidence": 0.9},
    ]
    prompt = _build_prompt("A. B.", {"words_per_minute": 120.0}, None, utterances)
    assert "address each speaker by their ID" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_llm_feedback.py -k wearer -v`
Expected: FAIL — `_build_prompt() got an unexpected keyword argument 'user_speaker'`.

- [ ] **Step 3: Implement the prompt changes**

In `src/services/llm.py`:

1. Add `user_speaker: Optional[int] = None,` to `generate_feedback`'s parameters (after `utterances`), and pass it through: `prompt = _build_prompt(transcript, features, emotion_scores, utterances, user_speaker=user_speaker)`.

2. Add the same parameter to `_build_prompt` (after `utterances`):

```python
def _build_prompt(
    transcript: str,
    features: Dict[str, Any],
    emotion_scores: Optional[Dict[str, float]],
    utterances: Optional[list[dict]] = None,
    user_speaker: Optional[int] = None,
) -> str:
```

3. Replace the multi-speaker section builder (`llm.py:234-253`) so wearer mode takes precedence:

```python
    # Build multi-speaker / wearer-focused section
    speaker_section = ""
    if utterances:
        unique_speakers = sorted({u["speaker"] for u in utterances if u.get("speaker") is not None})
        speaker_lines = []
        for u in utterances:
            spk = u.get("speaker")
            spk_label = f"Speaker {spk}" if spk is not None else "Unknown"
            speaker_lines.append(f"- {spk_label}: {u['text']}")
        speaker_block = "\n".join(speaker_lines)

        if user_speaker is not None:
            speaker_section = f"""
**Wearer-Focused Mode — coach ONLY Speaker {user_speaker}:**

The person you are coaching is Speaker {user_speaker} (identified by voice enrollment).
All other speakers are CONTEXT ONLY — use their words to understand what Speaker {user_speaker}
was responding to, but do NOT evaluate, score, or give feedback to them.
The computed features above were calculated only from Speaker {user_speaker}'s speech.
Every score and every piece of feedback must be about Speaker {user_speaker} alone.

Full conversation (for context):
```
{speaker_block}
```
"""
        elif len(unique_speakers) > 1:
            speaker_section = f"""
**Conversation Mode — Multi-Speaker Transcript:**

This recording has {len(unique_speakers)} distinct speakers. When giving feedback, address each speaker by their ID, comment on turn-taking, balance of speaking time, and how the speakers interact. Per-utterance breakdown:

```
{speaker_block}
```
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_llm_feedback.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/services/llm.py tests/test_llm_feedback.py
git commit -m "feat(llm): wearer-focused prompt — coach only the enrolled speaker"
```

---

### Task 4: Pipeline integration — attribution + user-only features

**Files:**
- Modify: `src/pipelines/analysis_pipeline.py` (`run_analysis_pipeline` `:165-252`, `_perform_analysis` `:34-129`)
- Test: `tests/test_pipeline_attribution.py` (new)

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_pipeline_attribution.py`:

```python
"""Pipeline branches for wearer-focused analysis. All services mocked."""
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.pipelines.analysis_pipeline import run_analysis_pipeline

STT_RESULT = {
    "text": "How was the demo? It went well, we shipped it.",
    "language": "en",
    "segments": [],
    "utterances": [
        {"speaker": 0, "text": "How was the demo?", "start": 0.0, "end": 1.5, "confidence": 0.9},
        {"speaker": 1, "text": "It went well, we shipped it.", "start": 1.6, "end": 3.9, "confidence": 0.9},
    ],
    "speakers": [0, 1],
}

MATCH = {"user_speaker": 1, "confidence": 0.81, "low_confidence": False}


def _run(tmp_path, *, profile, match, export_path="user.wav"):
    """Run the pipeline with all externals mocked; returns (results, mocks)."""
    combiner = MagicMock()
    combiner.combine_features.return_value = {"words_per_minute": 100.0}
    with (
        patch("src.pipelines.analysis_pipeline.stt.transcribe", return_value=dict(STT_RESULT)),
        patch("src.pipelines.analysis_pipeline.initialize_vector_store", return_value=None),
        patch("src.pipelines.analysis_pipeline.speaker_id.load_profile", return_value=profile),
        patch("src.pipelines.analysis_pipeline.speaker_id.match_user", return_value=match) as match_mock,
        patch("src.pipelines.analysis_pipeline.speaker_id.export_segments_wav", return_value=export_path) as export_mock,
        patch("src.pipelines.analysis_pipeline.FeatureCombiner", return_value=combiner),
        patch("src.pipelines.analysis_pipeline.emotion.analyze", return_value=None),
        patch("src.pipelines.analysis_pipeline.llm.generate_feedback", return_value="fb") as llm_mock,
    ):
        results = run_analysis_pipeline("audio.wav", output_dir=str(tmp_path))
    return results, {"combiner": combiner, "llm": llm_mock, "match": match_mock, "export": export_mock}


def _read_transcript(results):
    with open(results["transcript_path"], encoding="utf-8") as f:
        return json.load(f)


def test_no_profile_keeps_generic_behavior(tmp_path):
    results, mocks = _run(tmp_path, profile=None, match=None)
    saved = _read_transcript(results)
    assert saved["speaker_attribution"] == {"enabled": False, "reason": "no_profile"}
    mocks["match"].assert_not_called()
    # generic: full transcript text + full audio
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["transcript_text"] == STT_RESULT["text"]
    assert kwargs["audio_path"] == "audio.wav"
    assert mocks["llm"].call_args.kwargs.get("user_speaker") is None


def test_enrolled_match_runs_wearer_focused_analysis(tmp_path):
    results, mocks = _run(tmp_path, profile=np.ones(3), match=dict(MATCH))
    saved = _read_transcript(results)
    assert saved["speaker_attribution"]["enabled"] is True
    assert saved["speaker_attribution"]["user_speaker"] == 1
    # features computed on user-only text + user-only wav
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["transcript_text"] == "It went well, we shipped it."
    assert kwargs["audio_path"] == "user.wav"
    # LLM gets the FULL transcript (context) + user_speaker
    llm_kwargs = mocks["llm"].call_args.kwargs
    assert llm_kwargs["transcript"] == STT_RESULT["text"]
    assert llm_kwargs["user_speaker"] == 1


def test_match_failure_falls_back_to_generic(tmp_path):
    results, mocks = _run(tmp_path, profile=np.ones(3), match=None)
    saved = _read_transcript(results)
    assert saved["speaker_attribution"] == {"enabled": False, "reason": "match_failed"}
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["transcript_text"] == STT_RESULT["text"]
    assert kwargs["audio_path"] == "audio.wav"
    assert results["features_path"] is not None  # non-fatal: analysis completed


def test_export_failure_keeps_full_audio_but_user_text(tmp_path):
    results, mocks = _run(tmp_path, profile=np.ones(3), match=dict(MATCH), export_path=None)
    kwargs = mocks["combiner"].combine_features.call_args.kwargs
    assert kwargs["audio_path"] == "audio.wav"          # fell back to full audio
    assert kwargs["transcript_text"] == "It went well, we shipped it."  # text still user-only


def test_low_confidence_flag_persisted(tmp_path):
    low = {"user_speaker": 0, "confidence": 0.12, "low_confidence": True}
    results, _ = _run(tmp_path, profile=np.ones(3), match=low)
    saved = _read_transcript(results)
    assert saved["speaker_attribution"]["low_confidence"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_pipeline_attribution.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'speaker_id'` (not imported), then assertion failures.

- [ ] **Step 3: Implement the pipeline changes**

In `src/pipelines/analysis_pipeline.py`:

1. Import the service (with the other service imports, `:22`):

```python
from src.services import stt, emotion, llm, speaker_id
```

2. In `run_analysis_pipeline`, after `utterances = stt_result.get("utterances")` (`:201`) and **before** the transcript JSON is saved, insert:

```python
    # Wearer-focused attribution (optional, non-fatal): which speaker is the user?
    user_speaker: Optional[int] = None
    attribution: Dict[str, Any] = {"enabled": False, "reason": "no_profile"}
    profile = speaker_id.load_profile()
    if profile is not None:
        match = None
        try:
            match = speaker_id.match_user(audio_path, utterances or [], profile)
        except Exception as e:
            logger.warning(f"[SpeakerID] match_user raised (non-fatal): {e}")
        if match:
            attribution = {"enabled": True, **match}
            user_speaker = match["user_speaker"]
            logger.info(
                f"[SpeakerID] User is Speaker {user_speaker} "
                f"(confidence {match['confidence']}, low_confidence={match['low_confidence']})."
            )
        else:
            attribution = {"enabled": False, "reason": "match_failed"}
            logger.warning("[SpeakerID] Could not identify the user — generic analysis.")
    stt_result["speaker_attribution"] = attribution
```

(The transcript JSON dump at `:206-207` now persists the attribution automatically — `json.dump(stt_result, ...)` is unchanged.)

3. Thread `user_speaker` into the analysis thread. Change the `threading.Thread` args (`:217-221`):

```python
    analysis_thread = threading.Thread(
        target=_perform_analysis,
        args=(audio_path, transcript_text, utterances, output_dir_path, analysis_results, user_speaker),
        daemon=True,
    )
```

4. Update `_perform_analysis` to accept and use it. New signature:

```python
def _perform_analysis(
    audio_path_str: str,
    transcript_text: str,
    utterances: list[dict] | None,
    output_dir_path: Path,
    results_dict: dict,
    user_speaker: int | None = None,
) -> None:
```

Immediately after `feedback_out = ...` (`:56`), insert the user-only derivation:

```python
        # Wearer-focused mode: compute features on the user's speech only.
        # The LLM still receives the FULL audio + transcript as context.
        features_audio_path = audio_path_str
        features_text = transcript_text
        if user_speaker is not None and utterances:
            user_utts = [u for u in utterances if u.get("speaker") == user_speaker]
            if user_utts:
                features_text = " ".join(u["text"] for u in user_utts)
                user_wav = speaker_id.export_segments_wav(
                    audio_path_str,
                    [(u["start"], u["end"]) for u in user_utts],
                    str(output_dir_path / f"{base_name}_user.wav"),
                )
                if user_wav:
                    features_audio_path = user_wav
                else:
                    logger.warning(
                        "[Thread Analysis] User-only WAV export failed — "
                        "audio features fall back to the full recording."
                    )
```

Change the `combine_features` call (`:61-64`) to use them:

```python
            combined_features = combiner.combine_features(
                audio_path=features_audio_path,
                transcript_text=features_text,
            )
```

Change the `llm.generate_feedback` call (`:98-104`) to pass the wearer (full audio/transcript stay as-is — context!):

```python
        feedback_text = llm.generate_feedback(
            audio_path=audio_path_str,
            transcript=transcript_text,
            features=combined_features,
            emotion_scores=emotion_scores,
            utterances=utterances,
            user_speaker=user_speaker,
        )
```

(WPM correctness falls out for free: the user-only WAV's `duration` is the user's speaking time, and `word_count` comes from `features_text` — both user-only, so `words_per_minute` in `FeatureCombiner._derive_features_inline` is computed against user speaking time.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_pipeline_attribution.py tests/test_pipeline.py -v`
Expected: PASS (5 new + all existing pipeline tests — the trailing `user_speaker=None` default keeps old call sites working).

- [ ] **Step 5: Full backend suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/pipelines/analysis_pipeline.py tests/test_pipeline_attribution.py
git commit -m "feat(pipeline): wearer-focused analysis — user-only features, attribution in transcript"
```

---

### Task 5: Frontend — enroll API + EnrollmentPanel

**Files:**
- Modify: `frontend/src/api.ts` (transcript type `:9-17`; new functions at end)
- Create: `frontend/src/components/EnrollmentPanel.tsx`
- Modify: `frontend/src/components/VerbalVector.tsx` (render panel in input stage, after the Ray-Ban help block ending `:392`)
- Test: `frontend/src/components/EnrollmentPanel.test.tsx` (new)

- [ ] **Step 1: Extend api.ts types + functions**

In `frontend/src/api.ts`, add `speaker_attribution` to the transcript object type (inside the object branch of `UploadResponse['transcript']`, after `speakers?: number[];`):

```typescript
        speaker_attribution?: SpeakerAttribution;
```

And append at the end of the file:

```typescript
export type SpeakerAttribution =
  | { enabled: true; user_speaker: number; confidence: number; low_confidence: boolean }
  | { enabled: false; reason: 'no_profile' | 'match_failed' };

export interface EnrollmentStatus {
  created_at: string;
  duration_seconds: number;
  model: string;
}

export async function getEnrollment(): Promise<EnrollmentStatus | null> {
  const res = await client.get<EnrollmentStatus>('/api/enroll', {
    validateStatus: (s) => s === 200 || s === 404,
  });
  return res.status === 404 ? null : res.data;
}

export async function enrollVoice(audio: Blob, filename = 'enroll.webm'): Promise<EnrollmentStatus> {
  const formData = new FormData();
  formData.append('file', audio, filename);
  const res = await client.post<EnrollmentStatus>('/api/enroll', formData);
  return res.data;
}

export async function deleteEnrollment(): Promise<void> {
  await client.delete('/api/enroll', { validateStatus: (s) => s === 204 || s === 404 });
}
```

Note: `UploadResponse['transcript']` is a union with `string` — `SpeakerAttribution` must be declared before first use or hoisted via `export type`; TypeScript hoists type declarations, so appending is fine.

- [ ] **Step 2: Write failing EnrollmentPanel tests**

Create `frontend/src/components/EnrollmentPanel.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mockGetEnrollment = vi.fn();
const mockEnrollVoice = vi.fn();
const mockDeleteEnrollment = vi.fn();
vi.mock('../api', () => ({
  getEnrollment: () => mockGetEnrollment(),
  enrollVoice: (b: Blob) => mockEnrollVoice(b),
  deleteEnrollment: () => mockDeleteEnrollment(),
}));

import { EnrollmentPanel } from './EnrollmentPanel';

class MockMediaRecorder {
  static isTypeSupported() { return true; }
  static instances: MockMediaRecorder[] = [];
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  mimeType = 'audio/webm';
  state = 'recording';
  constructor() { MockMediaRecorder.instances.push(this); }
  start() {}
  stop() {
    this.ondataavailable?.({ data: new Blob(['x'], { type: 'audio/webm' }) });
    this.onstop?.();
  }
}

const STATUS = { created_at: '2026-06-12T00:00:00Z', duration_seconds: 31, model: 'ecapa' };

describe('EnrollmentPanel', () => {
  beforeEach(() => {
    mockGetEnrollment.mockReset();
    mockEnrollVoice.mockReset();
    mockDeleteEnrollment.mockReset();
    MockMediaRecorder.instances = [];
    (global as unknown as { MediaRecorder: unknown }).MediaRecorder = MockMediaRecorder;
    Object.defineProperty(global.navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }) },
    });
  });
  afterEach(() => {
    cleanup();
    delete (global as unknown as Record<string, unknown>).MediaRecorder;
  });

  it('shows not-enrolled state when GET returns null', async () => {
    mockGetEnrollment.mockResolvedValue(null);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => expect(screen.getByText(/not enrolled/i)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /record enrollment/i })).toBeInTheDocument();
  });

  it('shows enrolled state with re-record and delete', async () => {
    mockGetEnrollment.mockResolvedValue(STATUS);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => expect(screen.getByText(/voice enrolled/i)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /re-record/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
  });

  it('records and uploads an enrollment', async () => {
    mockGetEnrollment.mockResolvedValue(null);
    mockEnrollVoice.mockResolvedValue(STATUS);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => screen.getByRole('button', { name: /record enrollment/i }));
    await userEvent.click(screen.getByRole('button', { name: /record enrollment/i }));
    await waitFor(() => screen.getByRole('button', { name: /stop & save/i }));
    await userEvent.click(screen.getByRole('button', { name: /stop & save/i }));
    await waitFor(() => expect(mockEnrollVoice).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/voice enrolled/i)).toBeInTheDocument());
  });

  it('surfaces upload errors', async () => {
    mockGetEnrollment.mockResolvedValue(null);
    mockEnrollVoice.mockRejectedValue({ response: { data: { detail: 'Enrollment audio must be at least 10 seconds (got 3.0s).' } } });
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => screen.getByRole('button', { name: /record enrollment/i }));
    await userEvent.click(screen.getByRole('button', { name: /record enrollment/i }));
    await waitFor(() => screen.getByRole('button', { name: /stop & save/i }));
    await userEvent.click(screen.getByRole('button', { name: /stop & save/i }));
    await waitFor(() => expect(screen.getByText(/at least 10 seconds/i)).toBeInTheDocument());
  });

  it('deletes the enrollment', async () => {
    mockGetEnrollment.mockResolvedValue(STATUS);
    mockDeleteEnrollment.mockResolvedValue(undefined);
    render(<EnrollmentPanel />);
    await userEvent.click(screen.getByRole('button', { name: /enroll my voice/i }));
    await waitFor(() => screen.getByRole('button', { name: /delete/i }));
    await userEvent.click(screen.getByRole('button', { name: /delete/i }));
    await waitFor(() => expect(mockDeleteEnrollment).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(/not enrolled/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/EnrollmentPanel.test.tsx`
Expected: FAIL — module `./EnrollmentPanel` not found.

- [ ] **Step 4: Implement EnrollmentPanel**

Create `frontend/src/components/EnrollmentPanel.tsx`:

```typescript
import { useState, useRef, useEffect, type CSSProperties } from 'react';
import { ChevronRight, ChevronDown, Mic, Square, Trash2 } from 'lucide-react';
import { getEnrollment, enrollVoice, deleteEnrollment, type EnrollmentStatus } from '../api';

const boxStyle: CSSProperties = {
  marginTop: '0.5rem',
  padding: '0.75rem 1rem',
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  borderRadius: '0.5rem',
  color: '#475569',
  lineHeight: 1.5,
  textAlign: 'left',
};

const buttonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '0.375rem',
  padding: '0.375rem 0.75rem',
  borderRadius: '0.375rem',
  border: '1px solid #c7d2fe',
  background: '#eef2ff',
  color: '#4338ca',
  cursor: 'pointer',
  fontSize: '0.8125rem',
  marginRight: '0.5rem',
};

/**
 * One-time voice enrollment for wearer-focused analysis. Collapsible panel:
 * fetches status lazily on first expand, records ~30s via MediaRecorder,
 * uploads to /api/enroll.
 */
export function EnrollmentPanel() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<EnrollmentStatus | null | 'loading'>('loading');
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (!open || fetchedRef.current) return;
    fetchedRef.current = true;
    getEnrollment()
      .then(setStatus)
      .catch(() => { setStatus(null); setError('Could not reach the enrollment service.'); });
  }, [open]);

  useEffect(() => () => { streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];
      const rec = new MediaRecorder(stream);
      recorderRef.current = rec;
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => { void upload(); };
      rec.start(1000);
      setRecording(true);
    } catch {
      setError('Microphone access denied or unavailable.');
    }
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setRecording(false);
  };

  const upload = async () => {
    setBusy(true);
    try {
      const mime = recorderRef.current?.mimeType || 'audio/webm';
      const blob = new Blob(chunksRef.current, { type: mime });
      const updated = await enrollVoice(blob);
      setStatus(updated);
      setError(null);
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || 'Enrollment upload failed.');
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    try {
      await deleteEnrollment();
      setStatus(null);
      setError(null);
    } catch {
      setError('Could not delete the profile.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ width: '100%', fontSize: '0.875rem', marginBottom: '1.5rem' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', background: 'none', border: 'none', cursor: 'pointer', color: '#4f46e5', padding: 0, fontSize: '0.875rem' }}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        Enroll my voice (get feedback about you, not the whole room)
      </button>
      {open && (
        <div style={boxStyle}>
          {status === 'loading' && <p style={{ margin: 0 }}>Checking enrollment…</p>}
          {status !== 'loading' && status !== null && (
            <>
              <p style={{ margin: 0 }}>
                ✓ <strong>Voice enrolled</strong> ({status.duration_seconds.toFixed(0)}s sample
                , {new Date(status.created_at).toLocaleDateString()}). Conversations will be analyzed
                for <em>your</em> performance only.
              </p>
              <div style={{ marginTop: '0.5rem' }}>
                {!recording ? (
                  <button style={buttonStyle} disabled={busy} onClick={startRecording}>
                    <Mic size={14} /> Re-record
                  </button>
                ) : (
                  <button style={buttonStyle} disabled={busy} onClick={stopRecording}>
                    <Square size={14} /> Stop &amp; save
                  </button>
                )}
                <button style={buttonStyle} disabled={busy || recording} onClick={remove}>
                  <Trash2 size={14} /> Delete
                </button>
              </div>
            </>
          )}
          {status === null && (
            <>
              <p style={{ margin: 0 }}>
                <strong>Not enrolled.</strong> Record ~30 seconds of normal speech once; after that,
                conversation analyses focus on you and treat everyone else as context.
              </p>
              <div style={{ marginTop: '0.5rem' }}>
                {!recording ? (
                  <button style={buttonStyle} disabled={busy} onClick={startRecording}>
                    <Mic size={14} /> Record enrollment
                  </button>
                ) : (
                  <button style={buttonStyle} disabled={busy} onClick={stopRecording}>
                    <Square size={14} /> Stop &amp; save
                  </button>
                )}
              </div>
            </>
          )}
          {error && <p style={{ margin: '0.5rem 0 0', color: '#b91c1c' }}>{error}</p>}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Wire it into VerbalVector's input stage**

In `frontend/src/components/VerbalVector.tsx`: add `import { EnrollmentPanel } from './EnrollmentPanel';` with the other component imports, and render `<EnrollmentPanel />` immediately after the Ray-Ban help block's closing `</div>` (after `:392`, before the record/upload button grid).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/EnrollmentPanel.test.tsx && npx vitest run && npx tsc --noEmit`
Expected: 5 new pass; full suite green; no type errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api.ts frontend/src/components/EnrollmentPanel.tsx frontend/src/components/EnrollmentPanel.test.tsx frontend/src/components/VerbalVector.tsx
git commit -m "feat(frontend): voice enrollment panel + enroll API client"
```

---

### Task 6: ResultsDisplay — "You" labels + confidence banners

**Files:**
- Modify: `frontend/src/components/ResultsDisplay.tsx` (`renderTranscript` `:81-109`; banners near the top of the rendered main `:161-165`)
- Test: `frontend/src/components/ResultsDisplay.test.tsx` (append)

- [ ] **Step 1: Write failing tests**

Append to `frontend/src/components/ResultsDisplay.test.tsx` (reuse its `baseFeatures` fixture):

```typescript
describe('ResultsDisplay speaker attribution', () => {
  const utterances = [
    { speaker: 0, text: 'How was the demo?', start: 0.0, end: 1.5, confidence: 0.98 },
    { speaker: 1, text: 'It went well.', start: 1.6, end: 3.0, confidence: 0.97 },
  ];

  const withAttribution = (attribution: object) => ({
    message: 'ok',
    transcript: {
      text: 'How was the demo? It went well.',
      utterances,
      speakers: [0, 1],
      speaker_attribution: attribution,
    },
    features: baseFeatures,
    feedback: 'fb',
  });

  it('labels the matched speaker "You" and others "Speaker N"', () => {
    render(
      <ResultsDisplay
        analysisResult={withAttribution({ enabled: true, user_speaker: 1, confidence: 0.81, low_confidence: false })}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/You:/)).toBeInTheDocument();
    expect(screen.getByText(/Speaker 0:/)).toBeInTheDocument();
    expect(screen.queryByText(/Speaker 1:/)).not.toBeInTheDocument();
    // analysis-is-about-you note
    expect(screen.getByText(/about you/i)).toBeInTheDocument();
  });

  it('shows a low-confidence warning banner', () => {
    render(
      <ResultsDisplay
        analysisResult={withAttribution({ enabled: true, user_speaker: 0, confidence: 0.21, low_confidence: true })}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/weren't sure which speaker was you/i)).toBeInTheDocument();
  });

  it('shows a fallback note when attribution failed', () => {
    render(
      <ResultsDisplay
        analysisResult={withAttribution({ enabled: false, reason: 'match_failed' })}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.getByText(/couldn't identify you/i)).toBeInTheDocument();
  });

  it('renders nothing attribution-related when no_profile', () => {
    render(
      <ResultsDisplay
        analysisResult={withAttribution({ enabled: false, reason: 'no_profile' })}
        onAnalyzeAnother={() => {}}
        onNavigate={() => {}}
      />,
    );
    expect(screen.queryByText(/about you/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/couldn't identify you/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/You:/)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/ResultsDisplay.test.tsx`
Expected: the 4 new tests FAIL (no banners, "Speaker 1" still rendered).

- [ ] **Step 3: Implement**

In `frontend/src/components/ResultsDisplay.tsx`:

1. Update `renderTranscript` to label the user (replace `:81-109`):

```typescript
function renderTranscript(transcript: UploadResponse['transcript']): React.ReactElement {
    if (typeof transcript === 'string') {
        return <pre style={transcriptBoxStyle}>{transcript}</pre>;
    }
    if (!transcript) {
        return <pre style={transcriptBoxStyle}>Transcript not available.</pre>;
    }

    const { utterances, speakers, speaker_attribution: attribution } = transcript;
    const userSpeaker = attribution?.enabled ? attribution.user_speaker : null;

    if (utterances && utterances.length > 0 && speakers && speakers.length > 1) {
        return (
            <div style={transcriptBoxStyle}>
                {utterances.map((u, i) => (
                    <div key={i} style={{ marginBottom: '0.5rem' }}>
                        {u.speaker !== null && (
                            <strong>{u.speaker === userSpeaker ? 'You' : `Speaker ${u.speaker}`}: </strong>
                        )}
                        <span>{u.text}</span>
                    </div>
                ))}
            </div>
        );
    }

    return <pre style={transcriptBoxStyle}>{transcript.text || 'Transcript not available.'}</pre>;
}
```

2. In the component body, after the early return (`:127`), derive the attribution once:

```typescript
  const attribution =
    typeof analysisResult.transcript === 'object' && analysisResult.transcript
      ? analysisResult.transcript.speaker_attribution
      : undefined;
```

3. In the JSX, immediately after the `<h2>Analysis Results</h2>` (`:162-164`), add the banners:

```typescript
            {attribution?.enabled && !attribution.low_confidence && (
                <p style={{ textAlign: 'center', color: '#4338ca', marginTop: '-1.5rem', marginBottom: '2rem', fontSize: '0.875rem' }}>
                    This analysis is about you (your enrolled voice was matched in this recording).
                </p>
            )}
            {attribution?.enabled && attribution.low_confidence && (
                <div style={{ background: '#fffbeb', border: '1px solid #fde68a', color: '#92400e', borderRadius: '0.5rem', padding: '0.75rem 1rem', marginBottom: '2rem', fontSize: '0.875rem', textAlign: 'center' }}>
                    ⚠ We weren't sure which speaker was you — this analysis is our best guess. Verify the "You" labels in the transcript below.
                </div>
            )}
            {attribution && !attribution.enabled && attribution.reason === 'match_failed' && (
                <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', color: '#475569', borderRadius: '0.5rem', padding: '0.75rem 1rem', marginBottom: '2rem', fontSize: '0.875rem', textAlign: 'center' }}>
                    We couldn't identify you in this recording — showing a generic whole-recording analysis instead.
                </div>
            )}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/ResultsDisplay.test.tsx && npx vitest run && npx tsc --noEmit`
Expected: all pass (existing 6 + new 4), full suite green, types clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ResultsDisplay.tsx frontend/src/components/ResultsDisplay.test.tsx
git commit -m "feat(frontend): show 'You' labels and attribution confidence banners in results"
```

---

## Final Verification (after all tasks)

- [ ] Backend: `venv/bin/python -m pytest tests/ -q` — all green.
- [ ] Frontend: `cd frontend && npx vitest run && npx tsc --noEmit && npx eslint src` — green (the two pre-existing `no-explicit-any` errors in `api.ts:13,18` are known, not ours).
- [ ] **Local-only real-model smoke test** (needs network for the ~80MB model download, not CI):
  1. `venv/bin/python -c "from src.services.speaker_id import compute_embedding; print(compute_embedding('data/streams/<any-existing>.wav') is not None)"` → `True`.
  2. In the web app: enroll ~30s of your voice; upload a two-person recording; confirm results show "You" labels, the about-you note, and user-only metrics.
- [ ] Dispatch `pr-review-toolkit:code-reviewer` + `silent-failure-hunter` (the non-fatal fallback paths are the riskiest part) + `pr-test-analyzer`.
- [ ] Push branch, open PR (summary / test plan / rollback notes), run the PR-review sweep.

## Rollback

Fully additive. `DELETE /api/enroll` (or removing `data/voice_profile/`) restores byte-identical pipeline behavior; without a profile, no new code path executes. Frontend panel and banners render nothing in the no-profile state. Revert commits independently if needed — each task is self-contained.
