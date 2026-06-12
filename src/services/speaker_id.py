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
