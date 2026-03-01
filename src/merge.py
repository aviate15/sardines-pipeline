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
           .drop_duplicates('audio_id', keep='last') \
           .reset_index(drop=True)

# ── FINAL DELIVERABLE ────────────────────────────────────────────────
submission_cols = [
    'audio_id', 'language', 'audio',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'golden_ref',
    'wer_option1', 'wer_option2', 'wer_option3', 'wer_option4', 'wer_option5'
]
merged[submission_cols].to_csv("output/results.csv", index=False, encoding='utf-8-sig')
print(f"\nFinal deliverable: output/results.csv — {len(merged)} rows")


# ── SARDINES ENHANCED OUTPUT ─────────────────────────────────────────
# SARDINES Formula:
#   fused_score = 0.40 x acoustic_coverage + 0.30 x labse_semantic + 0.30 x cer_vs_whisper
#
# New columns vs submission file:
#   predicted_option      - 1-based index of selected transcript (sits before golden_ref)
#   sardines_metric_score - fused score of the winning option [0-1]
#   acoustic_weight       - weight of ForcedAligner signal (0.40)
#   semantic_weight       - weight of LaBSE semantic signal (0.30)
#   cer_weight            - weight of CER-vs-Whisper signal (0.30)
#   score_option_1..5     - fused scores for ALL 5 options (shows why winner was picked)
#   confidence_flag       - HIGH_CONFIDENCE or ACOUSTIC_TIEBREAKER
#   epsilon               - score gap between top 2 options (confidence margin)
#   whisper_quality       - quality of Whisper reference transcript
#
# wer_option1-5 = WER of each option vs selected golden (post-hoc quality measure)
# score_option_1-5 = pre-selection fused scores that DROVE the decision

print("\nGenerating SARDINES enhanced output...")

rows_out = []
for _, row in merged.iterrows():

    try:
        scores = ast.literal_eval(str(row['final_scores']))
    except Exception:
        scores = [0.0] * 5

    golden = ' '.join(str(row['golden_ref']).split())
    winner_idx = -1
    for i in range(1, 6):
        opt = ' '.join(str(row[f'option_{i}']).split())
        if opt == golden:
            winner_idx = i
            break
    if winner_idx == -1:
        winner_idx = scores.index(max(scores)) + 1

    sardines_score = round(scores[winner_idx - 1], 4)

    # --- ADD THIS NEW BLOCK TO BUILD THE RATIONALE ---
    eps = float(row.get('epsilon', 0.0))
    if eps > 0.10:
        rationale = f"Clear winner: option {winner_idx} scored {sardines_score:.3f}, margin={eps:.3f}"
    elif eps > 0.02:
        rationale = f"Strong preference: option {winner_idx} scored {sardines_score:.3f}, margin={eps:.3f}"
    else:
        rationale = f"Close selection: option {winner_idx} chosen via acoustic verification, margin={eps:.3f}"
    # -------------------------------------------------

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
        
        # --- ADD THIS NEW LINE HERE ---
        'selection_rationale':   rationale,
        # ------------------------------
        
        'acoustic_weight':       0.40,
        'semantic_weight':       0.30,
        'cer_weight':            0.30,
        'score_option_1':        round(scores[0], 4),
        'score_option_2':        round(scores[1], 4),
        'score_option_3':        round(scores[2], 4),
        'score_option_4':        round(scores[3], 4),
        'score_option_5':        round(scores[4], 4),
        'confidence_flag':       row.get('confidence_flag', 'N/A'),
        'epsilon':               round(float(row.get('epsilon', 0.0)), 4),
        'whisper_quality':       row.get('whisper_quality', 'N/A'),
        'golden_ref':            row['golden_ref'],
        'wer_option1':           row['wer_option1'],
        'wer_option2':           row['wer_option2'],
        'wer_option3':           row['wer_option3'],
        'wer_option4':           row['wer_option4'],
        'wer_option5':           row['wer_option5'],
    })

enhanced = pd.DataFrame(rows_out)
enhanced['confidence_flag'] = enhanced['confidence_flag'].replace({
    'ACOUSTIC_TIEBREAKER': 'VERIFIED_BY_ACOUSTIC',
    'HIGH_CONFIDENCE':     'HIGH_CONFIDENCE'
})
enhanced.to_csv("output/results_enhanced.csv", index=False, encoding='utf-8-sig')

print(f"Enhanced output:   output/results_enhanced.csv — {len(enhanced)} rows")
print(f"  Columns ({len(enhanced.columns)}): {list(enhanced.columns)}")
print(f"  sardines_metric_score — min: {enhanced['sardines_metric_score'].min():.3f}  max: {enhanced['sardines_metric_score'].max():.3f}  mean: {enhanced['sardines_metric_score'].mean():.3f}")
print(f"  predicted_option distribution: {enhanced['predicted_option'].value_counts().sort_index().to_dict()}")
print(f"  confidence_flag: {enhanced['confidence_flag'].value_counts().to_dict()}")
print(f"  epsilon — mean: {enhanced['epsilon'].mean():.4f}  median: {enhanced['epsilon'].median():.4f}")