"""
LLM feedback service using Gemini 2.5 Flash with native audio input.

Gemini receives the raw audio file alongside the transcript, Librosa
features, and Hume emotion scores — so it reasons about what it actually
hears, not just numbers on a page.

Returns a plain markdown string.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

NO_RELEVANT_CONTENT = "No relevant content was found in your stored transcripts."


def generate_feedback(
    audio_path: str,
    transcript: str,
    features: Dict[str, Any],
    emotion_scores: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Generate markdown feedback using Gemini 2.5 Flash with native audio.

    Args:
        audio_path:     Path to the original audio file (uploaded to Gemini).
        transcript:     Full text transcript from Deepgram.
        features:       Combined Librosa + text feature dict.
        emotion_scores: Hume emotion label → score dict (optional).

    Returns:
        Markdown-formatted feedback string, or None on error.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai is not installed. Run: pip install google-genai")
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

    if not Path(audio_path).is_file():
        logger.error(f"[LLM] Audio file not found: {audio_path}")
        return None

    logger.info(f"[LLM] Generating feedback with {GEMINI_LLM_MODEL}")

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Upload audio so Gemini can natively hear it
        logger.info("[LLM] Uploading audio to Gemini Files API...")
        audio_file = client.files.upload(file=audio_path)
        logger.info(f"[LLM] Audio uploaded: {audio_file.name}")

        prompt = _build_prompt(transcript, features, emotion_scores)

        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=[audio_file, prompt],
            config=types.GenerateContentConfig(
                temperature=0.6,
            ),
        )

        # Clean up the uploaded file after generation
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass  # Non-fatal

        feedback_text = response.text.strip() if response.text else None
        if not feedback_text:
            logger.error("[LLM] Gemini returned an empty response.")
            return None

        logger.info("[LLM] Feedback generated successfully.")
        return feedback_text

    except Exception as e:
        logger.error(f"[LLM] Gemini generation failed: {e}", exc_info=True)
        return None


def generate_rag_answer(
    query: str,
    context_chunks: list[dict],
) -> Optional[dict]:
    if not context_chunks:
        return {"answer": NO_RELEVANT_CONTENT}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai is not installed.")
        return None

    from config import GEMINI_API_KEY, GEMINI_LLM_MODEL

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set.")
        return None

    context_text = "\n\n".join(
        f"[{chunk.get('session_label', 'unknown')}] {chunk['text']}"
        for chunk in context_chunks
    )

    prompt = _build_rag_prompt(query, context_text)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return {"answer": response.text}
    except Exception as e:
        logger.error(f"[LLM] RAG generation failed: {e}", exc_info=True)
        return None


def _build_rag_prompt(query: str, context_text: str) -> str:
    return f"""You are a helpful assistant that answers questions about the user's recorded presentations and conversations. Use ONLY the transcript excerpts provided below. If the answer is not in the excerpts, say so — do not guess or fabricate information. When possible, mention which session the evidence comes from using the [session label] tags.

**Transcript Excerpts:**
{context_text}

**Question:** {query}"""


def _build_prompt(
    transcript: str,
    features: Dict[str, Any],
    emotion_scores: Optional[Dict[str, float]],
) -> str:
    """Build the Gemini prompt from transcript, extracted features, and optional Hume emotion scores."""

    features_json = json.dumps(features, indent=2, ensure_ascii=False, default=str)

    # Pre-format key metrics for inline references in the prompt
    wpm_val = features.get("words_per_minute", "N/A")
    wpm_str = f"{wpm_val:.1f} wpm" if isinstance(wpm_val, float) else str(wpm_val)

    pitch_std_val = features.get("pitch_std", "N/A")
    pitch_std_str = f"{pitch_std_val:.1f} Hz std" if isinstance(pitch_std_val, float) else str(pitch_std_val)

    volume_std_val = features.get("volume_std", "N/A")
    volume_std_str = f"{volume_std_val:.3f} std" if isinstance(volume_std_val, float) else str(volume_std_val)

    num_pauses_val = features.get("num_pauses", "N/A")
    num_pauses_str = str(num_pauses_val)

    filler_rate_val = features.get("filler_word_rate", "N/A")
    filler_rate_str = f"{filler_rate_val:.1f}%" if isinstance(filler_rate_val, float) else str(filler_rate_val)

    transcript_snippet = transcript[:75].replace("\n", " ").replace("'", "`") + "..."

    # Build emotion section if available
    emotion_section = ""
    if emotion_scores:
        top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        emotion_lines = "\n".join(f"  {name}: {score:.3f}" for name, score in top_emotions)
        emotion_section = f"""
**Vocal Emotion Scores (Hume Expression Measurement — top 10):**
```
{emotion_lines}
```
Use these scores to add qualitative nuance to your assessment of how the speaker sounds emotionally.
"""

    return f"""You are an expert communication coach. You have been given the speaker's audio recording to listen to directly, along with computed features and a transcript. Use EVERYTHING — what you hear AND the data — to give highly specific, evidence-based feedback.

**IMPORTANT:** Do not give generic advice. Every point must cite either a direct quote from the transcript, a specific feature value, or something you can hear in the audio.

**Presentation Transcript:**
```text
{transcript}
```

**Computed Acoustic & Text Features:**
```json
{features_json}
```
{emotion_section}
**Your Task:**
Generate high-value feedback following the Markdown structure below precisely.

**Output Format:**

## Overall Assessment
(Concise summary of effectiveness, key strengths and critical improvement areas, based on what you heard AND the data.)

## Key Scores (1-10)
*   **Clarity:** <score> (Reasoning: cite WPM, pauses, filler rate, lexical diversity, or transcript examples.)
*   **Engagement:** <score> (Reasoning: cite pitch/volume variation, WPM, or transcript examples.)
*   **Pacing:** <score> (Reasoning: **must reference WPM [{wpm_str}] and pause data [{num_pauses_str} pauses].**)
*   **Vocal Variety:** <score> (Reasoning: **must reference pitch [{pitch_std_str}] and volume [{volume_std_str}] variation.**)

## Strengths
(2-3 specific strengths with evidence from audio, transcript, or features.)
*   Strength 1: ... (Evidence: ...)
*   Strength 2: ... (Evidence: ...)

## Areas for Improvement
(2-3 actionable areas with specific evidence and concrete suggestions.)
*   Area 1: ... (Issue: ..., Evidence: filler rate is {filler_rate_str} / quote from transcript / what you heard, Suggestion: ...)
*   Area 2: ... (Issue: ..., Evidence: ..., Suggestion: ...)

## Content & Structure Analysis
(Based on the transcript, summarize the core message in 1-2 sentences, then analyse logical flow, transitions, organisation, and the effectiveness of the intro and conclusion. Quote specific examples.)
"""
