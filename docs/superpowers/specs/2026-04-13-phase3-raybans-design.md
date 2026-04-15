# Phase 3: Meta Ray-Bans Integration Design

> **Status:** Research complete, design proposed
> **Date:** 2026-04-13
> **Author:** Phase 3 research agent

---

## Research Findings

### SDK Availability

Meta does **not** provide a public developer SDK for the Ray-Ban Meta smart glasses. Key findings:

| Topic | Finding |
|-------|---------|
| **Public SDK** | None. No REST API, WebSocket API, or native SDK for third-party audio access. |
| **Meta View companion app** | Required for pairing, firmware, and Meta AI features. No developer API or hooks exposed. |
| **Project Aria (research program)** | Meta's Aria Research Kit (ARK) provides an SDK for the Project Aria research glasses, but this is a separate device program with restricted access for academic/research partners. It does **not** apply to consumer Ray-Ban Meta glasses. |
| **Meta Spark AR** | Focused on AR effects for Instagram/Facebook cameras. No integration with Ray-Ban Meta audio hardware. |
| **Open-source projects** | No known open-source libraries provide direct programmatic access to Ray-Ban Meta hardware. Community projects interact with the glasses only through standard Bluetooth audio protocols. |

### Audio Access Mechanism

The Ray-Ban Meta glasses function as a **standard Bluetooth audio device** when paired with a phone or computer:

- **Bluetooth profiles:** A2DP (stereo audio playback), HFP/HSP (hands-free profile for microphone input)
- **Bluetooth version:** 5.2 (supports LE Audio in newer firmware)
- **When paired with a computer:** The glasses appear as a standard audio input device in the OS sound settings. Browsers and applications can access the mic through standard APIs (`getUserMedia` in browsers, PyAudio/sounddevice in Python).
- **When paired with a phone:** Functions as a Bluetooth headset. Audio input is accessible to any app that uses the phone's standard microphone API.

### Audio Hardware Specs

| Spec | Value |
|------|-------|
| **Microphones** | 5-mic array (beamforming for voice isolation) |
| **Audio codecs** | AAC, SBC (standard Bluetooth audio codecs) |
| **Sample rate over Bluetooth** | 8 kHz (HFP narrowband), 16 kHz (mSBC wideband over HFP), or up to 32 kHz (LE Audio / LC3 codec on BT 5.2+). Not 48 kHz like a USB studio mic. |
| **Channels** | Mono (mic input over Bluetooth HFP) |
| **Latency** | ~100-200ms Bluetooth audio latency (acceptable for batch analysis). For streaming/live transcription, this compounds with Deepgram streaming latency (~300-500ms), yielding ~400-700ms total — see the streaming section. |

### Meta AI Voice Features (Not Accessible)

The "Hey Meta" voice assistant processes audio on-device and through Meta's cloud pipeline. This is a closed system:
- No way to intercept or redirect the Meta AI audio stream
- No webhooks, callbacks, or event APIs
- The Meta AI feature and VerbalVector would need to coexist as separate audio consumers (not a conflict since they use different activation mechanisms)

### Known Limitations

1. **Bluetooth HFP audio quality** is lower than direct USB mic recording (16 kHz mono vs 48 kHz stereo). This is a Bluetooth protocol constraint, not specific to the glasses.
2. **Simultaneous connections:** The glasses can pair with one device at a time for audio. If paired with a phone for calls, they cannot simultaneously stream mic audio to a laptop.
3. **Battery life:** ~4 hours of continuous use. Extended recording sessions may drain the glasses.
4. **No programmatic pairing:** The user must manually pair via Bluetooth settings. There is no API to initiate or manage the Bluetooth connection.
5. **getUserMedia device selection:** The user must explicitly select the Ray-Ban Meta as the audio input device in the browser's device picker or OS sound settings.
6. **Safari Bluetooth audio support:** Safari historically drops Bluetooth audio input devices from `enumerateDevices()` results. This can be a hard blocker for Safari users — Chrome is the recommended browser for this integration.

---

## Chosen Integration Path

### Path A: Bluetooth Mic (Selected)

**Rationale:** Meta provides no developer SDK or streaming API for the Ray-Ban Meta glasses. The glasses expose their microphone as a standard Bluetooth audio input device. This means:

- The existing browser-based mic recording (Phase 2B's `getUserMedia` flow) works **as-is** with the glasses
- The existing file upload flow works with any audio exported from the Meta View app
- **No backend changes are required** beyond what Phase 2A/2B already provide

**Path B (SDK Streaming) is not viable** -- there is no SDK to stream from.

**Path C (Companion App Relay) is a fallback** only for the case where a user cannot pair the glasses directly with their computer (e.g., mobile-only user). In that case, they would record via the Meta View app and upload the file.

---

## Architecture

### Audio Flow: Glasses to Pipeline

```
Ray-Ban Meta glasses
    |
    | Bluetooth HFP (8-16 kHz mono, up to 32 kHz with LE Audio)
    v
Computer / Phone (paired as audio input device)
    |
    | Browser: getUserMedia({audio: {deviceId: "ray-ban-device-id"}})
    | -- OR --
    | File upload: user records in Meta View app, exports, uploads
    v
VerbalVector Frontend (existing recording UI or upload form)
    |
    | POST /api/upload (multipart audio file)
    v
VerbalVector Backend (existing pipeline)
    |
    +-> Deepgram Nova-3 (batch STT) -- existing stt.py
    +-> Hume AI (emotion analysis) -- existing emotion.py
    +-> Gemini (LLM feedback) -- existing llm.py
    +-> ChromaDB (vector storage) -- existing/Phase 2A
```

### What Changes vs. Current System

| Component | Change Required | Notes |
|-----------|----------------|-------|
| Backend API (`api.py`) | **None** | Already accepts audio file uploads |
| STT service (`stt.py`) | **None** | Deepgram handles Bluetooth-quality audio fine |
| LLM service (`llm.py`) | **None** | Operates on transcript text, not raw audio |
| Emotion service (`emotion.py`) | **None** | Hume processes the audio file regardless of source |
| Frontend recording | **Minor** | Add device selector dropdown so user can pick the Ray-Ban mic |
| Frontend UX | **Minor** | Add guidance text for Bluetooth pairing setup |
| Config (`config.py`) | **None** | No new API keys or config values needed |

---

## Components

### New: Frontend Device Selector (Phase 3 implementation)

The browser `getUserMedia` API supports enumerating audio devices via `navigator.mediaDevices.enumerateDevices()`. The frontend should:

1. List available audio input devices in a dropdown
2. Let the user select the Ray-Ban Meta (it will appear with a Bluetooth device name like "Ray-Ban Meta" or similar)
3. Pass the selected `deviceId` to `getUserMedia`

```javascript
// Pseudocode for device enumeration
const devices = await navigator.mediaDevices.enumerateDevices();
const audioInputs = devices.filter(d => d.kind === 'audioinput');
// Populate dropdown with audioInputs
// On selection, use: getUserMedia({ audio: { deviceId: { exact: selectedDeviceId } } })
```

This is a **frontend-only change** and does not affect the backend.

### Existing: Batch STT (no changes)

The current `src/services/stt.py` uses Deepgram's pre-recorded (batch) API. This works well for the Bluetooth mic flow because:

- The user records a session via the browser
- The browser sends the complete audio file to the backend
- The backend transcribes the full file in one batch call

Bluetooth audio at 16 kHz mono is well within Deepgram Nova-3's supported input range. Deepgram handles sample rate conversion internally.

### Future: Streaming STT (not needed for Path A)

If a future iteration wants real-time feedback during recording (e.g., live transcript display), a streaming STT client would be needed. This would use Deepgram's WebSocket API (`deepgram.listen.live`). However, this is **not required for the initial integration** -- the current record-then-analyze flow is sufficient.

See "Streaming Pipeline" section below for the design if this is pursued later.

---

## Multi-Speaker Support

### Mic Characteristics

The Ray-Ban Meta's 5-mic array uses beamforming optimized for the wearer's voice. This means:

- **Presentation mode (wearer speaking):** Excellent capture quality. The beamforming isolates the wearer's voice from ambient noise.
- **Conversation mode (multiple speakers):** The wearer's voice will be captured clearly. Other speakers will be picked up but at lower quality/volume, depending on distance and ambient noise.

### Deepgram Diarization

Deepgram Nova-3 supports speaker diarization (`diarize=True`). This can be enabled in `stt.py` options to separate speakers in the transcript. This is a one-line change:

```python
options = PrerecordedOptions(
    model=DEEPGRAM_STT_MODEL,
    language="en",
    smart_format=True,
    filler_words=True,
    utterances=True,
    punctuate=True,
    diarize=True,  # <-- add this for multi-speaker
)
```

### Recommendation

- Start with single-speaker mode (the wearer). This is the primary use case for speech coaching/analysis.
- Multi-speaker diarization can be enabled as an option later for conversation analysis scenarios.

---

## Streaming Pipeline (Future / If Needed)

This section documents the design for a streaming STT pipeline, in case real-time transcription during recording is desired in a future phase. **This is not needed for the initial Path A integration.**

### Architecture

```
Browser (getUserMedia + MediaRecorder)
    |
    | WebSocket: ws://backend/api/stream
    | Sends audio chunks (e.g., 250ms intervals)
    v
FastAPI WebSocket endpoint (/api/stream)
    |
    | Forwards chunks to Deepgram
    v
Deepgram Live API (WebSocket)
    |
    | Returns partial + final transcripts
    v
Backend accumulates transcript
    |
    | Sends interim results back to browser via same WebSocket
    v
Browser displays live transcript
    |
    | On session end: backend runs full pipeline on accumulated transcript
    v
Standard pipeline (LLM feedback, emotion, storage)
```

### Deepgram Streaming Client

The Deepgram Python SDK supports live/streaming transcription:

```python
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

deepgram = DeepgramClient(api_key)
connection = deepgram.listen.live.v("1")

connection.on(LiveTranscriptionEvents.Transcript, handle_transcript)

options = LiveOptions(
    model="nova-3",
    language="en",
    encoding="linear16",
    sample_rate=16000,  # Match Bluetooth HFP
    channels=1,
    smart_format=True,
    filler_words=True,
    interim_results=True,
)

await connection.start(options)
# Send audio chunks: connection.send(audio_bytes)
# Finish: connection.finish()
```

### Key Considerations

- **Encoding:** Bluetooth HFP sends audio as PCM/SBC. The browser's MediaRecorder may re-encode. Use `linear16` (PCM 16-bit) for lowest latency, or `webm-opus` if MediaRecorder prefers that format.
- **Sample rate:** 8000 Hz for HFP narrowband, 16000 Hz for mSBC wideband, or up to 32000 Hz with LE Audio (LC3). Must match what the browser actually captures.
- **Interim vs final results:** Deepgram sends `is_final=False` for interim (partial) results and `is_final=True` for finalized segments. Only accumulate final results into the transcript.
- **Connection lifecycle:** One WebSocket connection per recording session. Close cleanly on session end.

---

## Testing Strategy

### Without Physical Glasses

Since testing requires actual Ray-Ban Meta hardware, the following strategies allow development and testing without the glasses:

1. **Any Bluetooth headset as proxy:** Pair any Bluetooth headset with the development machine. The audio flow is identical -- Bluetooth HFP mic input via `getUserMedia`. If it works with AirPods or any Bluetooth headset, it will work with Ray-Ban Meta.

2. **Mock audio source:** Use a virtual audio device (e.g., BlackHole on macOS, PulseAudio virtual sink on Linux) to simulate a Bluetooth audio input. Feed pre-recorded audio through the virtual device.

3. **Direct file upload:** Record audio on a phone paired with Ray-Ban Meta, transfer the file, and upload it through the existing file upload flow. This tests the full pipeline without needing browser Bluetooth pairing.

4. **Deepgram audio quality testing:** Submit 16 kHz mono audio files (simulating Bluetooth quality) to the existing batch STT pipeline to verify transcription accuracy is acceptable at lower sample rates.

### With Physical Glasses (Hardware Testing Checklist)

When Ray-Ban Meta hardware is available:

- [ ] Pair glasses with development laptop via Bluetooth
- [ ] Verify glasses appear as audio input device in OS sound settings
- [ ] Verify browser `enumerateDevices()` lists the glasses
- [ ] Record a 30-second sample via browser `getUserMedia` with glasses selected
- [ ] Submit recording through VerbalVector pipeline, verify transcript quality
- [ ] Test in quiet environment vs noisy environment
- [ ] Test wearer speaking vs conversation with another person
- [ ] Measure end-to-end latency (record -> upload -> transcript returned)
- [ ] Test battery drain during a 15-minute continuous recording session

---

## Open Questions

1. **Exact Bluetooth device name:** What name do the Ray-Ban Meta glasses advertise over Bluetooth? Needed for auto-detection in the device selector UI. (Requires hardware testing.)

2. **Browser compatibility:** Does `getUserMedia` with a Bluetooth audio device work reliably across Chrome, Firefox, Safari? Known issues with Safari and Bluetooth audio. (Requires testing.)

3. **Audio quality at 16 kHz:** Is Deepgram Nova-3 transcription accuracy materially worse at 16 kHz mono (Bluetooth HFP) compared to 48 kHz stereo (USB mic)? Preliminary expectation: Nova-3 handles this well, but needs quantitative comparison. (Can be tested without glasses using downsampled audio.)

4. **Wideband Bluetooth (mSBC vs LE Audio):** Do the Ray-Ban Meta glasses and the host OS negotiate mSBC wideband (16 kHz) or LE Audio / LC3 (up to 32 kHz) when available? This would improve audio quality. (Requires hardware testing.)

5. **Simultaneous Meta AI and recording:** Can the user have the "Hey Meta" AI active while simultaneously streaming mic audio to the browser? Or does Meta AI lock the audio input? (Requires hardware testing.)

6. **Continuous recording duration:** What is the practical maximum recording duration before Bluetooth disconnection or battery depletion? (Requires hardware testing.)

7. **MediaRecorder codec availability:** `MediaRecorder` codec support varies by browser — Firefox does not support `audio/webm;codecs=opus` the same way Chrome does. The streaming pipeline must check `MediaRecorder.isTypeSupported()` and fall back gracefully. (Can be tested without glasses.)

8. **Future SDK possibility:** Meta may release a developer SDK for Ray-Ban Meta in the future (especially as they push the AI glasses category). The architecture should be ready to add a streaming path if/when that happens. The streaming pipeline design in this doc serves as a blueprint.

---

## Implementation Roadmap

### Phase 3a: Minimal Integration (Frontend Only)

1. Add audio device selector dropdown to recording UI
2. Add pairing instructions/guidance for Bluetooth setup
3. Test with any Bluetooth headset as proxy
4. Estimated effort: 1-2 days

### Phase 3b: Audio Quality Optimization (If Needed)

1. Test Deepgram accuracy at 16 kHz mono vs 48 kHz stereo
2. If accuracy drops significantly, investigate:
   - Pre-processing: upsampling, noise reduction before STT
   - Deepgram options tuned for low-bandwidth audio
3. Estimated effort: 1-2 days

### Phase 3c: Streaming Pipeline (Future, If Needed)

1. Implement WebSocket endpoint `/api/stream`
2. Implement Deepgram streaming STT client
3. Add live transcript display in frontend
4. Estimated effort: 3-5 days

### Phase 3d: Multi-Speaker Support (Future)

1. Enable Deepgram diarization
2. Update transcript format to include speaker labels
3. Update LLM prompt to handle multi-speaker transcripts
4. Estimated effort: 2-3 days
