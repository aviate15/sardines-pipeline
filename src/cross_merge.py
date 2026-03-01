"""
cross_merge.py — Pull Machine A and Machine B slice files from the repo and
                 combine them into the final deliverable.

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
    1. Find both CSV files in the output/ folder
    2. Validate row counts and required columns
    3. Check for ID gaps or overlaps
    4. Merge and sort by audio_id
    5. Write output/results.csv          ← clean submission (exact brief format)
    6. Write output/results_enhanced.csv ← full debug output with all scores

════════════════════════════════════════════════════════════════════════════════
RULES FOR MACHINE A AND B
════════════════════════════════════════════════════════════════════════════════

  File naming MUST follow this exact pattern:   results_{START}_{END}.csv
  Machine A produces:                           results_1_50.csv
  Machine B produces:                           results_51_100.csv

  Both files must be placed in the output/ folder before running this script.
  The easiest way: both machines push to git, then pull on the merging machine.

  Git workflow:
    Machine A:  git add output/results_1_50.csv && git commit -m "A done" && git push
    Machine B:  git add output/results_51_100.csv && git commit -m "B done" && git push
    Either:     git pull && python cross_merge.py

════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import glob
import ast
import os
import sys

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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FIND SLICE FILES
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("  SARDINES — cross_merge.py")
print("=" * 60)
print()

slices = sorted(glob.glob("output/results_*_*.csv"))

# Exclude the final output files if they already exist from a previous run
slices = [s for s in slices if not s.endswith("results_enhanced.csv")]

if not slices:
    print("ERROR: No slice files found in output/")
    print()
    print("Expected files like:  output/results_1_50.csv")
    print("                      output/results_51_100.csv")
    print()
    print("Current contents of output/:")
    for f in sorted(glob.glob("output/*")):
        print(f"  {f}")
    print()
    print("Make sure both machines have pushed their files and you have pulled.")
    sys.exit(1)

print(f"[1/5] Found {len(slices)} slice file(s):")
for s in slices:
    print(f"      {s}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — VALIDATE EACH SLICE
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[2/5] Validating...")

all_ok      = True
all_ranges  = []  # list of (start, end, filepath)

for s in slices:
    basename = os.path.basename(s)              # results_1_50.csv
    stem     = basename.replace('.csv', '')     # results_1_50
    parts    = stem.replace('results_', '').split('_')  # ['1', '50']

    # Parse start/end from filename
    try:
        start, end = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        print(f"  FAIL  {s}")
        print(f"        Cannot parse start/end from filename.")
        print(f"        Rename it to:  results_START_END.csv  (e.g. results_1_50.csv)")
        all_ok = False
        continue

    d        = pd.read_csv(s, encoding='utf-8-sig')
    expected = end - start + 1

    # Row count check
    if len(d) != expected:
        print(f"  FAIL  {s}  —  {len(d)} rows found, expected {expected} ({start}–{end})")
        print(f"        Pipeline may have crashed. Re-run the missing rows before merging.")
        all_ok = False
    else:
        print(f"  OK    {s}  —  {len(d)} rows  (ids {start}–{end})")

    # Required columns check
    missing_cols = [c for c in REQUIRED_COLS if c not in d.columns]
    if missing_cols:
        print(f"  FAIL  {s}  —  missing columns: {missing_cols}")
        all_ok = False

    # audio_id range check — IDs in file should actually be within [start, end]
    actual_min = int(d['audio_id'].min())
    actual_max = int(d['audio_id'].max())
    if actual_min < start or actual_max > end:
        print(f"  WARN  {s}  —  audio_ids go from {actual_min} to {actual_max}, "
              f"expected {start}–{end}")

    all_ranges.append((start, end, s))

# Overlap / gap detection across slices
if len(all_ranges) >= 2:
    all_ranges_sorted = sorted(all_ranges, key=lambda x: x[0])
    print()
    for i in range(len(all_ranges_sorted) - 1):
        _, end_a, file_a = all_ranges_sorted[i]
        start_b, _, file_b = all_ranges_sorted[i + 1]
        if start_b <= end_a:
            print(f"  WARN  Overlap detected: {file_a} ends at {end_a}, "
                  f"{file_b} starts at {start_b}")
            print(f"        Duplicates will be removed (first occurrence kept).")
        elif start_b > end_a + 1:
            gap_ids = list(range(end_a + 1, start_b))
            print(f"  WARN  Gap detected: audio_ids {gap_ids} are missing.")
            print(f"        These rows are not covered by any machine.")
            all_ok = False

if not all_ok:
    print()
    print("Fix the errors above, then re-run cross_merge.py.")
    print("(Warnings are OK to proceed with — Errors are not.)")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMBINE AND SORT
# ─────────────────────────────────────────────────────────────────────────────

print()
print("[3/5] Combining and sorting...")

dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in slices]
merged = (
    pd.concat(dfs, ignore_index=True)
    .drop_duplicates(subset=['audio_id'], keep='first')
    .sort_values('audio_id')
    .reset_index(drop=True)
)

print(f"      {len(merged)} rows total, "
      f"audio_ids {int(merged['audio_id'].min())}–{int(merged['audio_id'].max())}")

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

    # Parse final_scores from string e.g. "[0.922, 0.786, 0.942, 0.922, 0.358]"
    try:
        scores = ast.literal_eval(str(row['final_scores']))
    except Exception:
        scores = [0.0] * 5

    # Find which option number (1-5) matches the golden_ref text
    golden     = ' '.join(str(row['golden_ref']).split())
    winner_idx = -1
    for i in range(1, 6):
        opt = ' '.join(str(row[f'option_{i}']).split())
        if opt == golden:
            winner_idx = i
            break
    if winner_idx == -1:
        # Fallback: highest fused score
        winner_idx = scores.index(max(scores)) + 1

    sardines_score = round(float(scores[winner_idx - 1]), 4)
    epsilon        = round(float(row.get('epsilon', 0.0)), 4)
    conf_flag      = str(row.get('confidence_flag', 'N/A'))
    w_quality      = str(row.get('whisper_quality', 'N/A'))

    # Human-readable rationale
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