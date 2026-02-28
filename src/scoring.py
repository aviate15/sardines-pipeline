import torch
import re
import math
from sentence_transformers import SentenceTransformer
from jiwer import wer as jwer, cer as jcer
from config import *


def load_labse():
    print("[LaBSE] Loading...")
    return SentenceTransformer(LABSE_MODEL)


def get_semantic_scores(norm_candidates, norm_ref, labse_model):
    if not norm_ref or not norm_ref.strip():
        return [0.5] * len(norm_candidates)

    safe = [c if c.strip() else "فارغ" for c in norm_candidates]

    # normalize_embeddings=True is REQUIRED for correct cosine similarity
    embs = labse_model.encode([norm_ref] + safe, normalize_embeddings=True)
    ref_emb = embs[0]

    # Cosine similarity is in [-1, 1] — shift to [0, 1]
    # Note: LaBSE cosine sims are typically 0.5–0.95 in practice,
    # so after shifting the effective range is ~0.75–0.97.
    # This compresses the semantic signal somewhat but does not break it.
    return [max(0.0, float(ref_emb @ e)) for e in embs[1:]]


def get_cer_scores(norm_candidates, norm_ref):
    if not norm_ref or not norm_ref.strip():
        return [0.5] * len(norm_candidates)

    scores = []
    for c in norm_candidates:
        if not c.strip():
            scores.append(0.0)
            continue
        try:
            scores.append(max(0.0, 1.0 - min(jcer(norm_ref, c), 1.0)))
        except Exception:
            scores.append(0.0)
    return scores


def normalize_per_sample(scores):
    # No scaling — raw scores fed directly into fusion.
    # Max-only scaling was still creating 1.0 ceiling ties on multiple options.
    # Raw acoustic [0.97,0.98,0.01] has real spread; after /max it becomes [0.99,1.0,0.01]
    # which collapses top candidates together and kills epsilon.
    return scores


def verbosity_penalty(candidate_text, all_texts):
    lengths = [len(t.replace(' ', '')) for t in all_texts if len(t.replace(' ', '')) > 50]
    if not lengths:
        return 1.0
    median_len = sorted(lengths)[len(lengths) // 2]
    candidate_len = len(candidate_text.replace(' ', ''))
    R = candidate_len / median_len if median_len > 0 else 1.0
    if R <= 1.5:
        return 1.0
    return math.exp(-(R - 1.5))


def format_penalty(raw_text):
    penalty = 1.0
    if re.search(r'\{[^}]*\}', str(raw_text)):
        penalty *= 0.95
    if re.search(r'(?:^|\n)\s*[\u0600-\u06FFa-zA-Z]{1,15}:', str(raw_text)):
        penalty *= 0.88
    return penalty


def fuse(A, S, C, whisper_quality):
    # A has already been penalized then normalized before this call.
    # Penalties are NOT re-applied here.
    # Whisper quality gate uses AND logic — see aligner_module.py.
    # Weights: 0.40 acoustic / 0.30 semantic / 0.30 CER
    use_whisper = whisper_quality in ("OK", "LOW_LOGPROB", "NO_SPEECH")

    final = []
    for i in range(len(A)):
        if use_whisper:
            score = W_ACOUSTIC * A[i] + W_SEMANTIC * S[i] + W_CER * C[i]
        else:
            # Whisper unreliable — pure acoustics only
            # Prevents LaBSE and CER from outvoting ForcedAligner 2-to-1
            # using a hallucinated or low-confidence reference
            score = A[i]
        final.append(score)
    return final


def confidence_check(scores, cer_scores=None, acoustic_scores=None):
    ranked     = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    winner_idx = ranked[0][0]
    epsilon    = ranked[0][1] - ranked[1][1]
    if epsilon == 0.0:
        max_score = ranked[0][1]
        tied = [i for i, s in enumerate(scores) if abs(s - max_score) < 0.0001]
        if len(tied) > 1 and cer_scores is not None:
            best_cer = max(cer_scores[i] for i in tied)
            cer_tied = [i for i in tied if abs(cer_scores[i] - best_cer) < 0.0001]
            if len(cer_tied) == 1:
                winner_idx = cer_tied[0]
            elif acoustic_scores is not None:
                winner_idx = max(cer_tied, key=lambda i: acoustic_scores[i])
        elif len(tied) > 1 and acoustic_scores is not None:
            winner_idx = max(tied, key=lambda i: acoustic_scores[i])
    return winner_idx, epsilon, epsilon >= CONFIDENCE_THRESHOLD


def acoustic_tiebreaker(A_raw, flags):
    """
    Tiebreaker when fused score margin is below CONFIDENCE_THRESHOLD.
    Receives A_raw (genuinely unpenalized scores).
    Applies penalties here fresh — this is intentional and correct.
    A_raw is pre-penalty in BOTH the main path and the tiebreaker path.
    In the main path:  penalize A_raw → A_penalized → normalize → fuse
    In the tiebreaker: penalize A_raw here → pick max
    One application of penalties in each path. Symmetric. Not a double-penalty.
    """
    # A_raw is always the pre-penalty scores — penalties applied fresh here,
    # not inherited from the main path. This is intentional for consistency.
    adjusted = []
    for i, a in enumerate(A_raw):
        if flags[i] == 'TRUNCATED':
            adjusted.append(a * 0.7)
        elif flags[i] == 'HEADER_LEAK':
            adjusted.append(a * 0.5)
        else:
            adjusted.append(a)
    return max(range(len(adjusted)), key=lambda i: adjusted[i])


def compute_wer_cer(candidate_norm, golden_norm):
    # WER argument order: reference first, hypothesis second.
    # jiwer.wer(reference, hypothesis) — denominator is reference word count.
    # Swapping these would use candidate length as denominator, giving wrong scores.
    if not golden_norm or not candidate_norm:
        return 1.0, 1.0
    try:
        w = round(min(jwer(golden_norm, candidate_norm), 1.0), 4)
        c = round(min(jcer(golden_norm, candidate_norm), 1.0), 4)
        return w, c
    except Exception:
        return 1.0, 1.0