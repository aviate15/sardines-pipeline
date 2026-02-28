import torch
import json
import os
import whisper
import torchaudio
from config import *


# ── DEVICE ──────────────────────────────────────────────────────────

def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── CACHE ────────────────────────────────────────────────────────────

def load_cache(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── CTC FALLBACK WRAPPER ─────────────────────────────────────────────
# If qwen-asr fails to load, we fall back to ctc-forced-aligner with
# wav2vec2-large-xlsr-53-arabic. The wrapper exposes the same .align()
# interface so get_alignment_score never needs to know which model it has.

class CTCFallbackAligner:
    """
    Wraps ctc-forced-aligner to expose the same .align(audio, text, language)
    interface as Qwen3ForcedAligner. get_alignment_score calls .align() and
    never needs to know which backend is running.
    """
    def __init__(self):
        print("[Aligner] Loading CTC fallback: wav2vec2-large-xlsr-53-arabic")
        from ctc_forced_aligner import (
            load_audio as ctc_load_audio,
            load_alignment_model,
            generate_emissions,
            get_alignments,
            get_spans,
            postprocess_results
        )
        # Store the ctc functions as instance attributes
        self._ctc_load_audio      = ctc_load_audio
        self._load_alignment_model = load_alignment_model
        self._generate_emissions  = generate_emissions
        self._get_alignments      = get_alignments
        self._get_spans           = get_spans
        self._postprocess_results = postprocess_results

        device = _device()
        self._model, self._tokenizer = load_alignment_model(
            "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
            device=device
        )
        self._device = device
        print("[Aligner] CTC fallback loaded OK")

    def align(self, audio, text, language):
        """
        Align text to audio using ctc-forced-aligner.
        Returns a list-of-lists structure matching the qwen output shape:
        [[segment, segment, ...]] where each segment has a .text attribute.
        Returns [[]] (empty inner list) if alignment produces no results —
        get_alignment_score treats that as 0.5 neutral.
        """
        try:
            audio_waveform = self._ctc_load_audio(audio)
            emissions, stride = self._generate_emissions(
                self._model, audio_waveform, self._device
            )
            tokens_starred, text_starred = self._tokenizer(
                text, separator="|"
            )
            segments, scores, blank_id = self._get_alignments(
                emissions, tokens_starred, self._tokenizer
            )
            spans = self._get_spans(tokens_starred, segments, blank_id)
            results = self._postprocess_results(text_starred, spans, stride, scores)

            if not results:
                return [[]]

            # Wrap in an object with .text so get_alignment_score can read it
            class _Seg:
                def __init__(self, t): self.text = t

            segments_out = [_Seg(r["label"]) for r in results if r.get("label")]
            return [segments_out] if segments_out else [[]]

        except Exception as e:
            print(f"[ERROR] CTCFallbackAligner.align: {e}")
            return [[]]


# ── LOAD ALIGNER ─────────────────────────────────────────────────────

def load_aligner():
    """
    Try Qwen3ForcedAligner first.
    If that fails, fall back to CTCFallbackAligner (wav2vec2-large-xlsr-53-arabic).
    If both fail, raise a clear error immediately — do not let the pipeline
    silently proceed with no acoustic signal.
    """
    device = _device()

    # ── PRIMARY: Qwen3ForcedAligner ──────────────────────────────────
    try:
        from qwen_asr import Qwen3ForcedAligner
        print(f"[Aligner] Loading Qwen3-ForcedAligner-0.6B on {device}")
        model = Qwen3ForcedAligner.from_pretrained(
            ALIGNER_MODEL_PATH,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device
        )
        
        print("[Aligner] Qwen3ForcedAligner loaded OK")
        return model

    except Exception as e:
        print(f"[WARN] Qwen3ForcedAligner failed to load: {e}")
        print("[Aligner] Attempting CTC fallback...")

    # ── FALLBACK: CTCFallbackAligner ─────────────────────────────────
    try:
        return CTCFallbackAligner()

    except Exception as e:
        print(f"[WARN] CTCFallbackAligner also failed to load: {e}")

    # ── BOTH FAILED: hard stop ────────────────────────────────────────
    # Do NOT let the pipeline continue with no acoustic signal.
    # Row 1 would get a neutral 0.5 for all 5 candidates, making the
    # acoustic signal completely useless and the selection random.
    raise RuntimeError(
        "\n\n"
        "FATAL: Both Qwen3ForcedAligner AND CTCFallbackAligner failed to load.\n"
        "The acoustic signal (60-100% weight) is unavailable.\n"
        "Running the pipeline without it produces meaningless results.\n\n"
        "Fix options:\n"
        "  1. pip install qwen-asr --upgrade\n"
        "  2. pip install ctc-forced-aligner --upgrade\n"
        "  3. Check VRAM — both models may OOM simultaneously, try WHISPER_MODEL='medium'\n"
        "  4. Screenshot this error and message the group immediately.\n"
    )


# ── WHISPER ───────────────────────────────────────────────────────────

def load_whisper():
    device = _device()
    print(f"[Whisper] Loading {WHISPER_MODEL} on {device}")
    # large-v3-turbo: ~1.5GB VRAM
    # If OOM: change WHISPER_MODEL to "medium" in config.py (do not commit)
    return whisper.load_model(WHISPER_MODEL, device=device)


def get_whisper_ref(audio_id, model, cache):
    key = f"w_{audio_id}"

    if key in cache:
        return cache[key]

    path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
    if not os.path.exists(path):
        result = {"ref_text": "", "quality": "NO_AUDIO",
                  "avg_logprob": -9.0, "no_speech_prob": 1.0}
        cache[key] = result
        save_cache(cache, WHISPER_CACHE)
        return result

    try:
        out = model.transcribe(
            path,
            language="ar",
            word_timestamps=False,
            fp16=torch.cuda.is_available()
        )

        segs = out.get("segments", [])

        if not segs:
            avg_lp, no_sp = -9.0, 1.0
        else:
            avg_lp = sum(s["avg_logprob"] for s in segs) / len(segs)
            no_sp  = sum(s.get("no_speech_prob", 0) for s in segs) / len(segs)

        # Quality gate uses AND logic:
        # Both conditions must be true to drop Whisper.
        # OR logic risks dropping good Whisper on Arabic clips with leading silence.
        # Tradeoff: AND is more permissive — if Whisper hallucinates with acceptable
        # logprob but high no_speech, the semantic signal stays. Monitor during run.
        quality = "OK"
        if avg_lp < LOGPROB_THRESHOLD and no_sp > NO_SPEECH_THRESHOLD:
            quality = "LOW_CONFIDENCE"
        elif avg_lp < LOGPROB_THRESHOLD:
            quality = "LOW_LOGPROB"
        elif no_sp > NO_SPEECH_THRESHOLD:
            quality = "NO_SPEECH"

        result = {
            "ref_text":       out["text"],
            "quality":        quality,
            "avg_logprob":    avg_lp,
            "no_speech_prob": no_sp
        }

    except Exception as e:
        print(f"[ERROR] Whisper {audio_id}: {e}")
        result = {"ref_text": "", "quality": "ERROR",
                  "avg_logprob": -9.0, "no_speech_prob": 1.0}

    cache[key] = result
    save_cache(cache, WHISPER_CACHE)
    return result


# ── ALIGNMENT SCORING ─────────────────────────────────────────────────

def get_alignment_score(audio_id, candidate_text, model, cache):
    """
    Score one candidate text against the audio using forced alignment.
    model can be Qwen3ForcedAligner or CTCFallbackAligner — both expose .align().
    Score = coverage ratio: aligned_chars / total_chars.
    Known limitation: biased toward shorter candidates with high coverage.
    Documented in technical report as a known limitation.
    """
    import hashlib
    key = f"a_{audio_id}_{hashlib.md5(candidate_text[:80].encode('utf-8')).hexdigest()[:8]}"

    if key in cache:
        return cache[key]

    path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
    if not os.path.exists(path) or not candidate_text.strip():
        cache[key] = 0.0
        save_cache(cache, ALIGNER_CACHE)
        return 0.0

    # Cap text length — TRUNCATED options are 32,767 chars long.
    # Passing that to the aligner creates a 35GB+ attention matrix → OOM.
    # 1000 chars covers ~100 Arabic words, more than any audio clip here.
    aligner_text = candidate_text[:1000]

    try:
        with torch.no_grad():
            results = model.align(
                audio=path,
                text=aligner_text,
                language="Arabic"
            )

        # Guard against both failure modes:
        # [] — model returned empty list (no alignment at all)
        # [[]] — model returned list containing empty list (aligned but no segments)
        # Both are treated as neutral 0.5 to avoid incorrectly eliminating
        # a valid candidate due to an aligner failure, not a text failure.
        if not results or not results[0]:
            print(f"[WARN] Aligner empty result id={audio_id} — returning 0.5 neutral")
            cache[key] = 0.5
            save_cache(cache, ALIGNER_CACHE)
            return 0.5

        segments = results[0]
        aligned_chars = sum(len(r.text) for r in segments)
        total_chars   = len(aligner_text.replace(' ', ''))
        score = min(1.0, aligned_chars / total_chars) if total_chars > 0 else 0.0
        score = max(0.0, min(1.0, score))

    except Exception as e:
        print(f"[ERROR] Aligner {audio_id}: {e}")
        score = 0.0

    cache[key] = score
    save_cache(cache, ALIGNER_CACHE)
    return score