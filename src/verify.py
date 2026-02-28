import pandas as pd

df = pd.read_csv('output/results.csv', encoding='utf-8-sig')
ok = True

def check(condition, message):
    global ok
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {message}")
    if not condition:
        ok = False

check(len(df) == 100, "100 rows present")

required_cols = [
    'audio_id', 'language', 'audio',
    'option_1', 'option_2', 'option_3', 'option_4', 'option_5',
    'golden_ref',
    'wer_option1', 'wer_option2', 'wer_option3', 'wer_option4', 'wer_option5'
]
check(all(c in df.columns for c in required_cols), "all required columns present")

# CER columns must NOT be in the final deliverable
cer_cols = [f'cer_option{i}' for i in range(1, 6)]
check(not any(c in df.columns for c in cer_cols), "CER columns correctly absent")

# Debug columns must NOT be in the final deliverable
for col in ['confidence_flag', 'epsilon', 'whisper_quality', 'final_scores']:
    check(col not in df.columns, f"debug column '{col}' correctly absent")

check(df['golden_ref'].isnull().sum() == 0, "no nulls in golden_ref")

wer_cols = [f'wer_option{i}' for i in range(1, 6)]
check(df[wer_cols].min().min() >= 0.0, "WER values >= 0")
check(df[wer_cols].max().max() <= 1.0, "WER values <= 1")

check(
    sorted(df['audio_id'].tolist()) == list(range(1, 101)),
    "audio_ids 1–100 all present"
)

# Exact schema check — no unexpected columns
check(
    set(df.columns) == set(required_cols),
    f"exact schema match (no unexpected columns)"
)

# Check golden candidate's own WER = 0.0
for i in range(1, 6):
    col_wer = f'wer_option{i}'
    col_opt = f'option_{i}'
    golden_rows = df[col_opt] == df['golden_ref']
    if golden_rows.any():
        check(df.loc[golden_rows, col_wer].eq(0.0).all(),
              f"golden WER = 0.0 when golden is option_{i}")

print()
if ok:
    print("ALL CHECKS PASSED. READY TO SUBMIT.")
else:
    print("FIX ERRORS BEFORE SUBMITTING.")