import pandas as pd
import glob

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

# Final deliverable — required columns only
# CER, confidence_flag, epsilon, final_scores are internal — dropped here
cols = [
    'audio_id', 'language', 'audio',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'golden_ref',
    'wer_option1', 'wer_option2', 'wer_option3', 'wer_option4', 'wer_option5'
]
merged[cols].to_csv("output/results.csv", index=False, encoding='utf-8-sig')
print(f"\nFinal deliverable: output/results.csv — {len(merged)} rows")