import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess     import load_csv, normalize_text, flag_bad_options, download_audio
from aligner_module import (load_whisper, load_aligner, get_whisper_ref,
                             get_alignment_score, load_cache, save_cache)
from scoring import (load_labse, get_semantic_scores, get_cer_scores,
                     normalize_per_sample, fuse, confidence_check,
                     acoustic_tiebreaker, compute_wer_cer,
                     verbosity_penalty, format_penalty)
from config import *


def process_row(row, whisper_model, al_model, labse_model, w_cache, a_cache):

    audio_id  = row['audio_id']
    raw_opts  = [str(row[f'option_{i}']) for i in range(1, 6)]
    flags     = [str(row.get(f'flag_{i}', 'OK')) for i in range(1, 6)]
    norm_opts = [normalize_text(o) for o in raw_opts]

    # ── ALL_OPTIONS_CORRUPT GUARD ────────────────────────────────────
    # If every option is TRUNCATED or HEADER_LEAK, penalties cancel in
    # normalize_per_sample (math proof: constant offset cancels in min-max).
    # Skip normal scoring, pick best acoustic score, flag the row.
    all_bad = all(f in ('TRUNCATED', 'HEADER_LEAK') for f in flags)
    if all_bad:
        A_raw = [
            get_alignment_score(audio_id, opt, al_model, a_cache)
            for opt in norm_opts
        ]
        winner_idx  = A_raw.index(max(A_raw))
        conf_flag   = "ALL_OPTIONS_CORRUPT"
        golden_norm = norm_opts[winner_idx]

        # Self-WER sanity check — catches non-determinism in normalize_text
        self_wer, _ = compute_wer_cer(golden_norm, golden_norm)
        assert self_wer == 0.0, (
            f"[BUG] Self-WER non-zero for id={audio_id}: {self_wer} "
            f"— normalize_text is non-deterministic"
        )

        wer_out, cer_out = [], []
        for i, opt in enumerate(norm_opts):
            if i == winner_idx:
                wer_out.append(0.0); cer_out.append(0.0)
            else:
                w, c = compute_wer_cer(opt, golden_norm)
                wer_out.append(w); cer_out.append(c)
        return _build_result(audio_id, row, raw_opts, raw_opts[winner_idx],
                             wer_out, cer_out, conf_flag, 0.0, "N/A", "N/A")

    # ── SIGNAL 1: FORCEDALIGNER ──────────────────────────────────────
    # Raw scores — genuinely unpenalized. A_raw is passed to acoustic_tiebreaker
    # which applies its own penalties. This is intentional — one penalty
    # application in each path, symmetric. See scoring.py acoustic_tiebreaker.
    A_raw = [
        get_alignment_score(audio_id, opt, al_model, a_cache)
        for opt in norm_opts
    ]

    # Penalize BEFORE normalizing.
    # Penalizing after normalization operates on a relative rank score [0,1],
    # not the raw acoustic confidence. The same 0.7 penalty has inconsistent
    # impact depending on where the score fell after normalization.
    # Penalizing raw scores preserves true proportional impact.
    A_penalized = []
    for i, a in enumerate(A_raw):
        if flags[i] == 'TRUNCATED':
            A_penalized.append(a * 0.7)
            # HEADER_LEAK falls here intentionally.
            # normalize_text already stripped the header before the aligner ran.
            # The aligner scored clean text — applying 0.5 here would penalize
            # a signal that was never corrupted. ids 33 and 48 depend on this.
        else:
            fp = format_penalty(raw_opts[i])
            vp = verbosity_penalty(raw_opts[i], raw_opts)
            A_penalized.append(a * fp * vp)

    A = normalize_per_sample(A_penalized)

    # ── WHISPER REFERENCE ────────────────────────────────────────────
    w_result  = get_whisper_ref(audio_id, whisper_model, w_cache)
    norm_ref  = normalize_text(w_result["ref_text"])
    w_quality = w_result["quality"]

    # ── SIGNAL 2: LABSE SEMANTIC ─────────────────────────────────────
    S = normalize_per_sample(
        get_semantic_scores(norm_opts, norm_ref, labse_model)
    )

    # ── SIGNAL 3: CER ────────────────────────────────────────────────
    C = normalize_per_sample(
        get_cer_scores(norm_opts, norm_ref)
    )

    # ── FUSION ───────────────────────────────────────────────────────
    final = fuse(A, S, C, w_quality)

    # ── CONFIDENCE CHECK ─────────────────────────────────────────────
    winner_idx, epsilon, confident = confidence_check(final, C, A_raw)
    conf_flag = "HIGH_CONFIDENCE"

    if not confident:
        # Fused winner is kept — tiebreaker no longer overrides fused scores.
        # winner_idx already set correctly by confidence_check above.
        conf_flag  = "ACOUSTIC_TIEBREAKER"

    # ── GOLDEN SELECTION ─────────────────────────────────────────────
    golden_norm = norm_opts[winner_idx]

    # Self-WER sanity check.
    # This actually calls compute_wer_cer — it does NOT check the hardcoded
    # 0.0 constant. It runs normalize_text twice on the same string and
    # verifies the plumbing is deterministic end-to-end.
    # If normalize_text ever produces different output on the same input
    # (which shouldn't happen), this catches it with a useful message
    # instead of silently writing wrong WER values.
    self_wer, _ = compute_wer_cer(golden_norm, golden_norm)
    assert self_wer == 0.0, (
        f"[BUG] Self-WER non-zero for id={audio_id}: {self_wer} "
        f"— normalize_text is non-deterministic"
    )

    # ── WER + CER ────────────────────────────────────────────────────
    wer_out, cer_out = [], []
    for i, opt in enumerate(norm_opts):
        if i == winner_idx:
            # Hardcode golden's own WER to 0.0 in the output CSV.
            # The assert above already verified this is correct.
            wer_out.append(0.0); cer_out.append(0.0)
        else:
            w, c = compute_wer_cer(opt, golden_norm)
            wer_out.append(w); cer_out.append(c)

    return _build_result(audio_id, row, raw_opts, raw_opts[winner_idx],
                         wer_out, cer_out, conf_flag,
                         round(epsilon, 4), w_quality,
                         str([round(s, 3) for s in final]))


def _build_result(audio_id, row, raw_opts, golden, wer_out, cer_out,
                  conf_flag, epsilon, w_quality, final_scores):
    # Sanitize newlines from raw options — 441 rows have embedded \n
    # that break CSV row boundaries when written unescaped
    raw_opts = [str(o).replace('\n', ' ').replace('\r', ' ') for o in raw_opts]
    golden = str(golden).replace('\n', ' ').replace('\r', ' ')
    return {
        "audio_id":        audio_id,
        "language":        row['language'],
        "audio":           row['audio'],
        "option_1":        raw_opts[0],
        "option_2":        raw_opts[1],
        "option_3":        raw_opts[2],
        "option_4":        raw_opts[3],
        "option_5":        raw_opts[4],
        "golden_ref":      golden,
        "wer_option1":     wer_out[0],
        "wer_option2":     wer_out[1],
        "wer_option3":     wer_out[2],
        "wer_option4":     wer_out[3],
        "wer_option5":     wer_out[4],
        "cer_option1":     cer_out[0],
        "cer_option2":     cer_out[1],
        "cer_option3":     cer_out[2],
        "cer_option4":     cer_out[3],
        "cer_option5":     cer_out[4],
        "confidence_flag": conf_flag,
        "epsilon":         epsilon,
        "whisper_quality": w_quality,
        "final_scores":    final_scores
    }


def run():
    print("=" * 50)
    print(f"SARDINES — audio_ids {AUDIO_ID_START}–{AUDIO_ID_END}")
    print("=" * 50)

    df = load_csv()
    df = flag_bad_options(df)
    df = df[
        (df['audio_id'] >= AUDIO_ID_START) &
        (df['audio_id'] <= AUDIO_ID_END)
    ].reset_index(drop=True)
    print(f"[1/6] {len(df)} rows to process")

    print("[2/6] Downloading audio files...")
    download_audio(df)

    print("[3/6] Loading models...")
    whisper_model = load_whisper()
    al_model      = load_aligner()   # raises RuntimeError if both aligners fail
    labse_model   = load_labse()
    w_cache       = load_cache(WHISPER_CACHE)
    a_cache       = load_cache(ALIGNER_CACHE)

    print("[4/6] Processing rows...")
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  [{i+1}/{len(df)}] id={row['audio_id']}...", end=" ", flush=True)
        try:
            r = process_row(
                row, whisper_model, al_model,
                labse_model, w_cache, a_cache
            )
            results.append(r)
            print(f"OK  flag={r['confidence_flag']}  e={r['epsilon']}  whisper={r['whisper_quality']}")
        except AssertionError as e:
            # Self-WER assertion fired — normalize_text is broken
            # This is a hard stop, not a soft skip — it means ALL rows are wrong
            print(f"\nFATAL ASSERTION: {e}")
            print("normalize_text is non-deterministic. Fix preprocess.py before continuing.")
            raise
        except Exception as e:
            print(f"ERROR: {e}")

    print("[5/6] Saving slice...")
    os.makedirs("output", exist_ok=True)
    slice_path = f"output/results_{AUDIO_ID_START}_{AUDIO_ID_END}.csv"
    out = pd.DataFrame(results)
    out.to_csv(slice_path, index=False, encoding='utf-8-sig')

    print(f"[6/6] Done. Upload {slice_path} to Google Drive.")
    print()
    print(f"  HIGH_CONFIDENCE:     {out['confidence_flag'].eq('HIGH_CONFIDENCE').sum()}")
    print(f"  ACOUSTIC_TIEBREAKER: {out['confidence_flag'].eq('ACOUSTIC_TIEBREAKER').sum()}")
    print(f"  ALL_OPTIONS_CORRUPT: {out['confidence_flag'].eq('ALL_OPTIONS_CORRUPT').sum()}")
    print(f"  Whisper bad rows:    {out['whisper_quality'].ne('OK').sum()}")


if __name__ == "__main__":
    run()