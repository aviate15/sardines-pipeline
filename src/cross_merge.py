"""
cross_merge.py — Combine Machine A (rows 1-50) and Machine B (rows 51-100)
                 into the final deliverable.

════════════════════════════════════════════════════════════════════════════════
THE FULL WORKFLOW (read this once)
════════════════════════════════════════════════════════════════════════════════

  Machine A runs pipeline.py for rows 1–50.
  When done, it pushes  output/results_1_50.csv  to the shared git repo.

  Machine B runs pipeline.py for rows 51–100.
  When done, it pushes  output/results_51_100.csv  to the shared git repo.

  Once BOTH files are in the repo, run THIS script on either machine:

      python cross_merge.py

  It will:
    1. Load exactly results_1_50.csv and results_51_100.csv — nothing else
    2. Validate row counts and required columns
    3. Merge and sort by audio_id
    4. Write output/results.csv          ← clean submission (exact brief format)
    5. Write output/results_enhanced.csv ← full debug output with all scores

════════════════════════════════════════════════════════════════════════════════
RULES FOR MACHINE A AND B
════════════════════════════════════════════════════════════════════════════════

  Machine A produces:   output/results_1_50.csv    (exactly 50 rows)
  Machine B produces:   output/results_51_100.csv  (exactly 50 rows)

  Git workflow:
    Machine A:  git add output/results_1_50.csv && git commit -m "A done" && git push
    Machine B:  git add output/results_51_100.csv && git commit -m "B done" && git push
    Either:     git pull && python cross_merge.py

════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import ast
import os
import sys

# ── HARDCODED: only ever these two files ─────────────────────────────────────
SLICE_A = "output/results_1_50.csv"      # Machine A — rows 1-50
SLICE_B = "output/results_51_100.csv"    # Machine B — rows 51-100

REQUIRED_COLS = [
    'audio_id', 'language', 'audio',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'golden_ref',
    'wer_option1', 'wer_option2', 'wer_option3', 'wer_option4', 'wer_option5',
    'final_scores', 'confidence_flag', 'epsilon', 'whisper_quality'
]

SUBMISSION_COLS = [
    'audio_id', 'language', 'audio',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'golden_ref',
    'wer_option1', 'wer_option2', 'wer_option3', 'wer_option4', 'wer_option5'
]

print()
print("=" * 60)
print("  SARDINES — cross_merge.py")
print("=" * 60)
print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CHECK BOTH FILES EXIST
# ─────────────────────────────────────────────────────────────────────────────

print("[1/5] Checking for slice files...")

missing = [f for f in [SLICE_A, SLICE_B] if not os.path.exists(f)]
if missing:
    print()
    print("ERROR: Missing required file(s):")
    for f in missing:
        print(f"  {f}")
    print()
    print("Machine A must push:  output/results_1_50.csv")
    print("Machine B must push:  output/results_51_100.csv")
    print("Then run:  git pull && python cross_merge.py")
    sys.exit(1)

print(f"      ✓  {SLICE_A}")
print(f"      ✓  {SLICE_B}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — VALIDATE EACH FILE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[2/5] Validating...")

all_ok = True

checks = [
    (SLICE_A, 1,  50),
    (SLICE_B, 51, 100),
]

for filepath, expected_start, expected_end in checks:
    d        = pd.read_csv(filepath, encoding='utf-8-sig')
    expected = expected_end - expected_start + 1

    # Row count
    if len(d) != expected:
        print(f"  FAIL  {filepath}  —  {len(d)} rows found, expected {expected}")
        print(f"        Pipeline may have crashed mid-run. Re-run missing rows first.")
        all_ok = False
    else:
        print(f"  OK    {filepath}  —  {len(d)} rows  (ids {expected_start}–{expected_end})")

    # Required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in d.columns]
    if missing_cols:
        print(f"  FAIL  {filepath}  —  missing columns: {missing_cols}")
        all_ok = False

    # audio_id range sanity check
    actual_min = int(d['audio_id'].min())
    actual_max = int(d['audio_id'].max())
    if actual_min < expected_start or actual_max > expected_end:
        print(f"  WARN  {filepath}  —  audio_ids run {actual_min}–{actual_max}, "
              f"expected {expected_start}–{expected_end}")

if not all_ok:
    print()
    print("Fix the errors above, then re-run cross_merge.py.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMBINE AND SORT
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[3/5] Combining and sorting...")

df_a = pd.read_csv(SLICE_A, encoding='utf-8-sig')
df_b = pd.read_csv(SLICE_B, encoding='utf-8-sig')

merged = (
    pd.concat([df_a, df_b], ignore_index=True)
    .sort_values('audio_id')
    .reset_index(drop=True)
)

print(f"      {len(merged)} rows total  "
      f"(ids {int(merged['audio_id'].min())}–{int(merged['audio_id'].max())})")

if len(merged) != 100:
    print(f"  WARN  Expected 100 rows total, got {len(merged)}")
if sorted(merged['audio_id'].tolist()) != list(range(1, 101)):
    print(f"  WARN  audio_ids are not a clean 1–100 sequence")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WRITE results.csv (clean submission file)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[4/5] Writing submission file...")

os.makedirs("output", exist_ok=True)
merged[SUBMISSION_COLS].to_csv("output/results.csv", index=False, encoding='utf-8-sig')
print(f"      ✓  output/results.csv  ({len(merged)} rows, {len(SUBMISSION_COLS)} columns)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — WRITE results_enhanced.csv (full debug output)
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[5/5] Writing enhanced file...")

rows_out = []

for _, row in merged.iterrows():

    try:
        scores = ast.literal_eval(str(row['final_scores']))
    except Exception:
        scores = [0.0] * 5

    golden     = ' '.join(str(row['golden_ref']).split())
    winner_idx = -1
    for i in range(1, 6):
        opt = ' '.join(str(row[f'option_{i}']).split())
        if opt == golden:
            winner_idx = i
            break
    if winner_idx == -1:
        winner_idx = scores.index(max(scores)) + 1

    sardines_score = round(float(scores[winner_idx - 1]), 4)
    epsilon        = round(float(row.get('epsilon', 0.0)), 4)
    conf_flag      = str(row.get('confidence_flag', 'N/A'))
    w_quality      = str(row.get('whisper_quality', 'N/A'))

    if conf_flag == 'HIGH_CONFIDENCE':
        rationale = (f"Strong preference: option {winner_idx} scored "
                     f"{sardines_score}, margin={epsilon:.3f}")
    elif epsilon == 0.0:
        rationale = (f"Tied scores: option {winner_idx} chosen via "
                     f"acoustic tiebreaker (A_raw)")
    else:
        rationale = (f"Close call: option {winner_idx} chosen via "
                     f"acoustic verification, margin={epsilon:.3f}")

    rows_out.append({
        'audio_id':              row['audio_id'],
        'language':              row['language'],
        'audio':                 row['audio'],
        'option_1':              row['option_1'],
        'option_2':              row['option_2'],
        'option_3':              row['option_3'],
        'option_4':              row['option_4'],
        'option_5':              row['option_5'],
        'predicted_option':      winner_idx,
        'sardines_metric_score': sardines_score,
        'selection_rationale':   rationale,
        'acoustic_weight':       0.40,
        'semantic_weight':       0.30,
        'cer_weight':            0.30,
        'score_option_1':        round(float(scores[0]), 4),
        'score_option_2':        round(float(scores[1]), 4),
        'score_option_3':        round(float(scores[2]), 4),
        'score_option_4':        round(float(scores[3]), 4),
        'score_option_5':        round(float(scores[4]), 4),
        'confidence_flag':       conf_flag,
        'epsilon':               epsilon,
        'whisper_quality':       w_quality,
        'golden_ref':            row['golden_ref'],
        'wer_option1':           row['wer_option1'],
        'wer_option2':           row['wer_option2'],
        'wer_option3':           row['wer_option3'],
        'wer_option4':           row['wer_option4'],
        'wer_option5':           row['wer_option5'],
    })

enhanced = pd.DataFrame(rows_out)
enhanced.to_csv("output/results_enhanced.csv", index=False, encoding='utf-8-sig')
print(f"      ✓  output/results_enhanced.csv  "
      f"({len(enhanced)} rows, {len(enhanced.columns)} columns)")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("  DONE")
print("=" * 60)
print()
print(f"  Rows processed:      {len(merged)}")
print(f"  Audio IDs:           {int(merged['audio_id'].min())}–{int(merged['audio_id'].max())}")
print()

print("  Confidence breakdown:")
for flag, n in enhanced['confidence_flag'].value_counts().items():
    print(f"    {flag:<35} {n} rows")

print()
print("  Epsilon (score margin between top 2 options):")
print(f"    Exactly 0.0 (tie):   {(enhanced['epsilon'] == 0.0).sum()} rows")
print(f"    Mean:                {enhanced['epsilon'].mean():.4f}")

print()
print("  Which option was picked most:")
for opt, n in enhanced['predicted_option'].value_counts().sort_index().items():
    print(f"    Option {opt}:  {n} times")

print()
print("  ─────────────────────────────────────")
print("  Submit:  output/results.csv")
print("  ─────────────────────────────────────")
print()