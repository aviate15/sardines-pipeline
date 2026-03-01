import pandas as pd
import glob
import ast

slices = sorted(glob.glob("output/results_*_*.csv"))
print(f"Found slices: {slices}")

all_ok = True
for s in slices:
    d = pd.read_csv(s, encoding='utf-8-sig')
    parts = s.replace('.csv', '').replace('output/results_', '').replace('output\\results_', '').split('_')
    start, end = int(parts[0]), int(parts[1])
    expected = end - start + 1
    if len(d) == expected:
        print(f"  OK: {s} ({len(d)} rows)")
    else:
        print(f"  WARNING: {s} has {len(d)} rows, expected {expected}")
        all_ok = False

if not all_ok:
    print("\nFix missing rows before continuing.")
    exit(1)

dfs    = [pd.read_csv(f, encoding='utf-8-sig') for f in slices]
merged = pd.concat(dfs, ignore_index=True) \
           .sort_values('audio_id') \
           .reset_index(drop=True)

# ── FINAL DELIVERABLE ────────────────────────────────────────────────
# Required columns only — no internal debug columns
cols = [
    'audio_id', 'language', 'audio',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'golden_ref',
    'wer_option1', 'wer_option2', 'wer_option3', 'wer_option4', 'wer_option5'
]
merged[cols].to_csv("output/results.csv", index=False, encoding='utf-8-sig')
print(f"\nFinal deliverable: output/results.csv — {len(merged)} rows")


# ── SARDINES ENHANCED OUTPUT ─────────────────────────────────────────
# Adds predicted_option, sardines_metric_score, signal weights,
# cer_score, confidence_flag, and epsilon for reporting/presentation.
# Formula: sardines_score = 0.40 * acoustic + 0.30 * semantic + 0.30 * CER
# final_scores column contains the per-option fused scores from this formula.

print("\nGenerating enhanced output with SARDINES metric columns...")

rows_out = []
for _, row in merged.iterrows():

    # Parse fused scores list e.g. [0.867, 0.944, 0.418, 0.882, 0.867]
    try:
        scores = ast.literal_eval(str(row['final_scores']))
    except Exception:
        scores = [0.0] * 5

    # Identify which option was selected (1-based) by matching golden_ref text
    golden = ' '.join(str(row['golden_ref']).split())
    winner_idx = -1
    for i in range(1, 6):
        opt = ' '.join(str(row[f'option_{i}']).split())
        if opt == golden:
            winner_idx = i
            break

    # Fallback: pick highest fused score if text match fails
    if winner_idx == -1:
        winner_idx = scores.index(max(scores)) + 1

    # Sardines metric score = fused score of the winning option
    sardines_score = round(scores[winner_idx - 1], 4)

    # CER score for the winning option (1 − CER vs Whisper reference)
    # cer_option columns store raw CER (0=perfect, 1=worst)
    # so cer_score = 1 - cer_option gives the signal contribution [0,1]
    cer_raw = float(row.get(f'cer_option{winner_idx}', 0.0))
    cer_score = round(max(0.0, 1.0 - cer_raw), 4)

    rows_out.append({
        'audio_id':              row['audio_id'],
        'language':              row['language'],
        'audio':                 row['audio'],
        'option_1':              row['option_1'],
        'option_2':              row['option_2'],
        'option_3':              row['option_3'],
        'option_4':              row['option_4'],
        'option_5':              row['option_5'],
        'predicted_option':      winner_idx,           # 1-based index of selected transcript
        'sardines_metric_score': sardines_score,       # fused score of winner (0–1)
        'acoustic_weight':       0.40,                 # ForcedAligner signal weight
        'semantic_weight':       0.30,                 # LaBSE semantic signal weight
        'cer_weight':            0.30,                 # CER signal weight
        'cer_score':             cer_score,            # CER signal value for winner (0–1)
        'confidence_flag':       row.get('confidence_flag', 'N/A'),
        'epsilon':               row.get('epsilon', 0.0),
        'golden_ref':            row['golden_ref'],    # full text of selected transcript
        'wer_option1':           row['wer_option1'],
        'wer_option2':           row['wer_option2'],
        'wer_option3':           row['wer_option3'],
        'wer_option4':           row['wer_option4'],
        'wer_option5':           row['wer_option5'],
    })

enhanced = pd.DataFrame(rows_out)
enhanced.to_csv("output/results_enhanced.csv", index=False, encoding='utf-8-sig')
print(f"Enhanced output:   output/results_enhanced.csv — {len(enhanced)} rows")
print(f"  Columns: {list(enhanced.columns)}")
print(f"  sardines_metric_score range: {enhanced['sardines_metric_score'].min():.3f} – {enhanced['sardines_metric_score'].max():.3f}")
print(f"  predicted_option distribution: {enhanced['predicted_option'].value_counts().sort_index().to_dict()}")