# Wearer-Focused Analysis — Design

**Date:** 2026-06-11
**Status:** Approved in discussion — pending spec review
**Branch:** `feat/wearer-focused-analysis`

## Vision (and where this fits)

The end goal is a hands-free coaching loop: "Hey Siri, analyze this conversation"
(AirPods) → record → VerbalVector analyzes → feedback read back. That splits into
two sub-projects:

1. **This spec — the backend "brain":** analyze *the user's* performance in any
   recording, using everyone else's speech as context but never as the subject.
   Buildable and testable today against the existing web frontend.
2. **Phase 2 (separate spec): the iOS Siri client.** A thin native app over the
   API this spec produces. Out of scope here.

Personal tool first; design choices must not wall off a future multi-user
product, but nothing multi-user is built now.

## Problem

VerbalVector currently analyzes *the whole recording*: features (WPM, fillers,
clarity, expressiveness) are computed over all speech, and feedback coaches
"the speaker" — meaningless in a multi-person conversation. The user wants:

- **Metrics and coaching about themselves only.**
- **Full conversational context preserved** — the LLM must see the whole
  diarized transcript so feedback understands what the user was responding to.
- Works for monologues trivially (user is the only speaker).

The missing capability: knowing **which diarized speaker is the user**.
Deepgram diarizes (Speaker 0/1/2…) but does not identify. Decision: **voice
enrollment** — a one-time voiceprint, matched against each diarized speaker per
recording.

## Decisions made (with the user)

| Question | Decision |
|---|---|
| Subject of analysis | Always the user; others' speech is context only |
| Speaker identification | One-time voice enrollment (voiceprint matching) |
| Ambiguous/failed match | **Best guess + flag** — always complete, surface low confidence prominently |
| Enrollment UX | Web app section: record ~30s in browser, re-recordable |
| Embedding tech | **SpeechBrain ECAPA-TDNN, local** (PyTorch already in the stack; voice never leaves the machine) |
| Sequencing | Backend brain now; iOS client is phase 2 |
| Tenancy | Single profile (personal tool) |

## Architecture

One new service module + threading "who is the user" through the existing
pipeline. CLAUDE.md rules hold:

- `src/services/speaker_id.py` is the **single wrapper** for the embedding
  model (swap providers by editing one file).
- All knobs in **config.py** with env fallbacks.
- Speaker ID is an **optional, non-fatal service**: any failure → log warning,
  fall back to today's generic whole-recording analysis, flag it in the
  response. STT and LLM remain pipeline-fatal, unchanged.

### New config (config.py)

```python
SPEAKER_EMBED_MODEL = os.environ.get("SPEAKER_EMBED_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
SPEAKER_MATCH_THRESHOLD = float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.30"))  # cosine sim; below = low confidence
VOICE_PROFILE_DIR = ROOT_DIR / "data" / "voice_profile"
ENROLL_MIN_SECONDS = float(os.environ.get("ENROLL_MIN_SECONDS", "10"))
```

### Component 1 — `src/services/speaker_id.py`

Wraps SpeechBrain ECAPA-TDNN (lazy-loaded singleton, CPU is fine).

- `compute_embedding(audio_path, segments=None) -> np.ndarray | None`
  Embedding of the audio (optionally only the given `(start, end)` segments).
- `match_user(audio_path, utterances, profile_embedding) -> dict | None`
  For each diarized speaker: concatenate up to ~30s of that speaker's segments
  (from utterance timestamps), embed, cosine-similarity vs the profile. Returns
  `{"user_speaker": int, "confidence": float, "low_confidence": bool}` —
  always the best-scoring speaker; `low_confidence=True` when the best score is
  below `SPEAKER_MATCH_THRESHOLD`. Returns `None` on any failure (caller falls
  back to generic).

Both functions may return `None` or raise; callers treat either as service
failure (per testing rules, both paths get tests).

### Component 2 — Enrollment API (api.py)

- `POST /api/enroll` — multipart audio file. Validates extension/size like
  upload; requires ≥ `ENROLL_MIN_SECONDS` of audio, else `400` with a clear
  message. Computes embedding, saves `VOICE_PROFILE_DIR/embedding.npy` +
  `profile.json` ({created_at, duration_seconds, model}). Overwrites any
  existing profile (re-enrollment). Returns profile metadata.
- `GET /api/enroll` — `200` with profile metadata, or `404` if none.
- `DELETE /api/enroll` — removes the profile; `204`.

Single profile, no auth (personal tool on a private machine — same trust model
as every existing endpoint).

### Component 3 — Pipeline integration (analysis_pipeline.py)

After STT returns the diarized transcript:

1. **No profile on disk** → today's behavior exactly. (Also the natural
   monologue path when unenrolled.)
2. **Profile exists** → `match_user(...)`:
   - **Success** → wearer-focused mode:
     - **Text features** computed on the user's utterances only; WPM uses the
       user's speaking time (sum of their utterance durations), not total
       recording duration.
     - **Audio features**: concatenate the user's segments into a temporary
       user-only WAV; run the existing audio feature extraction on it
       (reuses code; monologue degenerates to whole-file, same as today).
     - **Feedback prompt**: full diarized transcript as context + explicit
       instruction — "coach only Speaker N (the user); treat all other
       speakers as context, do not evaluate them."
   - **Failure (`None`/raise)** → generic analysis + warning log + attribution
     disabled flag (below). Non-fatal, mirrors the Hume pattern.

Speaker attribution result is written **into the transcript JSON** as
`speaker_attribution`. This makes it flow through every existing surface for
free — `/api/upload` response, WS `session_end`, and
`GET /api/sessions/{id}/result` all read the same files via
`read_analysis_results`; no handler changes.

```json
"speaker_attribution": {
  "enabled": true,
  "user_speaker": 1,
  "confidence": 0.74,
  "low_confidence": false
}
// or, when fallback occurred / no profile:
"speaker_attribution": { "enabled": false, "reason": "no_profile" | "match_failed" }
```

Vector store: unchanged (full transcript stored as today). YAGNI.

Live `/api/stream` sessions inherit all of this for free — they call the same
pipeline and read the same files.

### Component 4 — Web frontend

- **Enrollment panel** on the input page (collapsible, like the Ray-Ban help):
  shows enrollment status (`GET /api/enroll`), records ~30s in-browser
  (MediaRecorder — capture code already exists), uploads to `POST /api/enroll`,
  offers re-record / delete.
- **ResultsDisplay**: when `speaker_attribution.enabled`, label the user's
  utterances **"You"** (others stay "Speaker N"); show a prominent banner when
  `low_confidence` ("We weren't sure which speaker was you — verify below");
  small note when `enabled: false` with a profile present ("couldn't identify
  you — generic analysis shown").
- **api.ts**: `enroll*` functions + `speaker_attribution` added to the
  transcript type.

## Error handling

| Case | Behavior |
|---|---|
| Low-confidence match | Best-guess analysis + `low_confidence: true` + visible UI banner |
| speaker_id service crash / unreadable profile | Generic analysis + `{enabled:false, reason:"match_failed"}` + warning log |
| No profile | Generic analysis + `{enabled:false, reason:"no_profile"}` (silent — this is the default state) |
| Enrollment audio too short / silent / bad format | `400` with actionable message |
| Single-speaker recording, enrolled | Matches trivially; features = whole recording (same numbers as today) |
| STT / LLM failure | Unchanged: pipeline-fatal, 500 |

## Testing

Mock SpeechBrain everywhere (no model download in CI; mirror the
Deepgram/Hume mocking pattern):

- **speaker_id unit tests**: matching picks the highest-similarity speaker
  (synthetic embeddings); below-threshold → `low_confidence`; segment
  concatenation respects the ~30s cap; returns `None` on internal failure.
- **Enrollment API**: 200 happy path (files written), 400 too-short, 404/200
  GET, 204 DELETE, overwrite on re-enroll.
- **Pipeline branches**: no-profile = byte-identical behavior to today;
  enrolled+match → user-only features and context-bearing prompt (assert the
  prompt contains other speakers' text AND the coach-only-Speaker-N
  instruction); enrolled+service-failure → generic + `enabled:false` (assert
  non-fatal); WPM uses user speaking time.
- **Frontend**: enrollment panel states (none/enrolled/error), ResultsDisplay
  "You" labeling + low-confidence banner + fallback note.
- **Local-only smoke test** (not CI): real model, two-voice recording, assert
  the right speaker wins.

## Out of scope (phase 2+)

- iOS Siri client (own spec/plan/build cycle).
- Multi-user profiles, accounts, auth.
- On-device/streaming speaker ID (match runs post-recording only).
- Re-architecting emotion/vector services.

## Rollback

Additive throughout. Delete the profile (or `DELETE /api/enroll`) and the
pipeline is byte-identical to today; the service module and endpoints are
unused without a profile. Frontend panel hides when the API 404s.
