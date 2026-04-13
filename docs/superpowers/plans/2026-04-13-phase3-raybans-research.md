# Phase 3: Meta Ray-Bans Integration Research Plan

> **For agentic workers:** This is a research-first plan. The goal is to produce a concrete integration spec, not to ship code. Code tasks are exploratory prototypes, not production features.

**Goal:** Investigate Meta Ray-Bans SDK capabilities, determine how audio reaches the backend, and produce a design document that Phase 3 implementation can follow. Also prototype the WebSocket streaming endpoint if the SDK supports real-time audio.

**Architecture question to answer:** Do the glasses expose a standard Bluetooth mic (browser can capture via `getUserMedia`), or does Meta's SDK provide a streaming API that needs a custom WebSocket endpoint?

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `docs/superpowers/specs/2026-04-13-phase3-raybans-design.md` | Create | Integration design doc — the main deliverable |
| `src/services/streaming_stt.py` | Create (prototype) | WebSocket Deepgram streaming client if needed |
| `tests/test_streaming_stt.py` | Create (prototype) | Basic test for streaming client |

---

### Task 1: Research Meta Ray-Bans SDK and audio capabilities

**Goal:** Determine the exact mechanism by which audio from the Ray-Bans can be accessed programmatically.

- [ ] **Step 1: Research Meta SDK documentation**

Search for and read:
- Meta Aria SDK / Meta Ray-Bans developer documentation
- Meta Spark AR (if relevant to Ray-Bans audio)
- Meta's "Research Kit" for Ray-Ban Stories / Ray-Ban Meta
- Any open-source projects that interface with Ray-Bans audio

Key questions to answer:
1. What SDK does Meta provide for the current generation Ray-Ban Meta glasses?
2. Does the SDK allow direct audio streaming from the glasses to a custom backend?
3. Or do the glasses pair via Bluetooth as a standard audio device, making them accessible through the browser's `getUserMedia` API?
4. Is there a REST API, WebSocket API, or native SDK (Android/iOS only)?
5. What audio formats and sample rates does the glasses' mic output?
6. Are there latency constraints to be aware of?
7. Does the Meta View app (companion app) expose any developer APIs?

- [ ] **Step 2: Document findings**

Create a research notes section in the design doc covering:
- SDK name and version
- Audio access mechanism (Bluetooth mic vs SDK stream vs companion app relay)
- Supported platforms (web, iOS, Android, desktop)
- Audio format specs
- Any rate limits or authentication requirements
- Known limitations

- [ ] **Step 3: Determine integration path**

Based on findings, classify into one of three paths:

**Path A: Bluetooth Mic (simplest)**
The glasses act as a standard Bluetooth audio device. The browser's `getUserMedia` picks them up like any other mic. Phase 2B's recording feature works as-is. No backend changes needed beyond what Phase 2A/2B provide.

**Path B: SDK Streaming API**
Meta provides an API that streams audio chunks. We need:
- A WebSocket endpoint (`/api/stream`) on the backend
- A Deepgram streaming STT client (not batch)
- Real-time pipeline: chunks → streaming STT → accumulate transcript → store on session end

**Path C: Companion App Relay**
Audio is only accessible through the Meta View companion app. We'd need to either:
- Build a mobile integration (out of scope for now)
- Find a way to relay audio from the companion app to our backend
- Use the companion app's export feature and upload the resulting file (falls back to existing upload flow)

---

### Task 2: Write integration design document

**Files:**
- Create: `docs/superpowers/specs/2026-04-13-phase3-raybans-design.md`

Based on Task 1 findings, write a design doc covering:

- [ ] **Step 1: Write the design doc**

Structure:

```markdown
# Phase 3: Meta Ray-Bans Integration Design

## Research Findings
(Summary of SDK capabilities, audio access mechanism, platform support)

## Chosen Integration Path
(A, B, or C — with rationale)

## Architecture
(How audio flows from glasses → backend → pipeline)

## Components
(What needs to be built, modified, or stays as-is)

## Multi-Speaker Support
(Conversation mode vs presentation mode — does the mic capture multiple speakers?)

## Streaming Pipeline (if Path B)
(WebSocket endpoint, Deepgram streaming STT, real-time transcript accumulation)

## Testing Strategy
(How to test without physical glasses — mock audio source, simulated streams)

## Open Questions
(Anything unresolved that needs hardware testing)
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-04-13-phase3-raybans-design.md
git commit -m "docs: add Phase 3 Meta Ray-Bans integration design

Research findings on SDK capabilities, chosen integration path,
architecture for audio flow from glasses to pipeline."
```

---

### Task 3: Prototype streaming STT client (if Path B)

**Skip this task if Task 1 determines Path A or C.**

If the Ray-Bans SDK provides streaming audio, we need a Deepgram streaming client alongside the existing batch client.

- [ ] **Step 1: Research Deepgram streaming SDK**

Check `deepgram-sdk` for WebSocket/streaming support:
- `deepgram.listen.live` or similar
- Required options for streaming (encoding, sample_rate, channels)
- How to handle partial vs final transcripts

- [ ] **Step 2: Create streaming_stt.py prototype**

Create `src/services/streaming_stt.py`:

```python
"""
Streaming STT service using Deepgram Nova-3 (live/WebSocket).

Prototype for Phase 3 Ray-Bans integration. This module handles
real-time audio chunks and accumulates transcript segments.
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """Manages a streaming Deepgram connection for real-time STT."""

    def __init__(self, on_transcript: Callable[[str, bool], None]):
        """
        Args:
            on_transcript: Callback(text, is_final) called for each transcript segment.
        """
        self.on_transcript = on_transcript
        self._connection = None
        self._full_transcript = ""

    async def start(self) -> None:
        """Open the Deepgram streaming connection."""
        # TODO: Implement with deepgram.listen.live
        raise NotImplementedError("Streaming STT is a Phase 3 prototype")

    async def send_audio(self, chunk: bytes) -> None:
        """Send an audio chunk to the streaming connection."""
        raise NotImplementedError("Streaming STT is a Phase 3 prototype")

    async def stop(self) -> str:
        """Close the connection and return the full accumulated transcript."""
        raise NotImplementedError("Streaming STT is a Phase 3 prototype")

    @property
    def transcript(self) -> str:
        return self._full_transcript
```

- [ ] **Step 3: Create test stub**

Create `tests/test_streaming_stt.py`:

```python
"""Placeholder tests for streaming STT — to be implemented with the full prototype."""
import pytest


def test_streaming_transcriber_import():
    """Verify the module can be imported."""
    from src.services.streaming_stt import StreamingTranscriber
    assert StreamingTranscriber is not None


@pytest.mark.asyncio
async def test_streaming_transcriber_not_implemented():
    """Start should raise NotImplementedError until fully implemented."""
    from src.services.streaming_stt import StreamingTranscriber

    transcriber = StreamingTranscriber(on_transcript=lambda text, final: None)
    with pytest.raises(NotImplementedError):
        await transcriber.start()
```

- [ ] **Step 4: Commit**

```bash
git add src/services/streaming_stt.py tests/test_streaming_stt.py
git commit -m "feat(stt): add streaming STT prototype for Phase 3

Skeleton StreamingTranscriber class with start/send_audio/stop interface.
Not yet implemented — raises NotImplementedError. Tests verify import
and expected behavior."
```

---

## Summary

| Task | Deliverable | What |
|------|------------|------|
| 1 | Research notes | SDK capabilities, audio access mechanism, integration path decision |
| 2 | Design doc | Full Phase 3 integration spec |
| 3 | Prototype code (conditional) | Streaming STT client skeleton if SDK provides streaming audio |

Task 1 is pure research — web searches, documentation reading, open-source project analysis. Task 2 depends on Task 1 findings. Task 3 is conditional on the integration path chosen.

## Key Constraint

This worktree should NOT modify any existing production files. Its deliverables are documentation and prototypes only. Production changes happen after Phase 2A and 2B land and this design is reviewed.
