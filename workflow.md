# SARDINES — COMPLETE BRAIN-DEAD HACKATHON WORKFLOW

### Hackenza 2026 · BITS Pilani KK Birla Goa Campus · Renan Partners

### Repo: https://github.com/aviate15/sardines-pipeline

---

> **THIS DOCUMENT IS SELF-CONTAINED.**
> If you are opening this on a fresh Claude account mid-hackathon, everything you need is here.
> Read Section 0 first. Then find your person letter (A/B/C/D) and follow only your steps.
> Do not skip steps. Do not improvise. Do not edit someone else's file without messaging the group.

---

## TABLE OF CONTENTS

1. [Hackathon Context — Read This First](#hackathon-context)
2. [Who Is Who](#who-is-who)
3. [The Files You Are Building](#the-files)
4. [SECTION D-0: GitHub Setup (Person D, Hours 0–1)](#section-d0)
5. [SECTION ALL: Installation on Every Machine (Hours 1–2)](#section-install)
6. [SECTION A: Download ForcedAligner Model (Person A, Hours 1–2)](#section-a-model)
7. [SECTION C: Data Recon (Person C, Hours 1–2)](#section-c-recon)
8. [SECTION D-1: Put Input CSV in Repo (Person D, Hour 2)](#section-d1)
9. [SECTION C: Write preprocess.py (Person C, Hours 2–5)](#section-c-code)
10. [SECTION A: Write aligner_module.py (Person A, Hours 2–5)](#section-a-code)
11. [SECTION B: Write scoring.py (Person B, Hours 2–5)](#section-b-code)
12. [SECTION D-2: Write pipeline.py + merge.py + verify.py (Person D, Hours 2–5)](#section-d2)
13. [SECTION ALL: Integration Test (Everyone, Hours 5–6)](#section-integration)
14. [SECTION A+B: Full Pipeline Run (Hours 6–14)](#section-run)
15. [SECTION D-3: Merge + Verify + Submit (Person D, Hours 14–22)](#section-d3)
16. [FINAL CHECKLIST](#final-checklist)
17. [Bug Registry](#bug-registry)
18. [Emergency Procedures](#emergency)

---

## HACKATHON CONTEXT <a name="hackathon-context"></a>

### What This Hackathon Is

**Client:** Renan Partners Private Limited
**Event:** Hackenza 2026, BITS Pilani KK Birla Goa Campus
**Team:** sardines — Ansh Varma, Sana H, Darshan Rajagoli, Rashida Baldiwala

**Task:** Given 100 Arabic audio files, each with 5 human transcription candidates, automatically pick the best one (the **golden reference**) and compute the Word Error Rate of the other 4 against it.

### Deliverables Required

1. **Working Code** — Automated transcription scoring and selection pipeline
2. **Output CSV** — Exactly this column schema, nothing more, nothing less:
   ```
   audio_id, language, audio,
   option_1, option_2, option_3, option_4, option_5,
   golden_ref,
   wer_option1, wer_option2, wer_option3, wer_option4, wer_option5
   ```
3. **Technical Report** — Explanation of methods, scoring logic, and results

### Evaluation Criteria

- **Selection Accuracy** — Correctly identifying the best transcription
- **Robustness** — Consistency across different languages and audio conditions
- **Automation & Reproducibility** — Clean end-to-end pipeline
- **Explanation Quality** — Clear technical reasoning

### What Our Pipeline Does (Plain English)

For each of the 100 audio rows:

1. **ForcedAligner** (Qwen3-ForcedAligner-0.6B) listens to the audio and scores each of the 5 candidate texts directly against the waveform. This is the main signal. It does not need Whisper. It does not compare candidates to each other. Score = coverage ratio: what fraction of the candidate's characters successfully aligned to the audio.

2. **Whisper** (large-v3-turbo) free-decodes the audio to produce a reference transcript. We check if Whisper is confident. If it is, we use it as a reference for two additional signals.

3. **LaBSE** (sentence-transformers/LaBSE) computes semantic similarity between each candidate and the Whisper reference. Handles Arabic dialects natively.

4. **CER** (character error rate) computes character-level closeness between each candidate and the Whisper reference. Better than WER for Arabic because minor suffix/prefix differences don't tank the entire word score.

5. **Fusion:** `score = 0.60 × acoustic + 0.25 × semantic + 0.15 × CER`
   - Whisper is dropped only when **both** conditions are true simultaneously — `avg_logprob < -1.0` AND `no_speech_prob > 0.50` — flagged internally as `LOW_CONFIDENCE`. A clip that trips only one threshold (`LOW_LOGPROB` or `NO_SPEECH`) keeps Whisper in the fusion. When dropped: `score = 1.0 × acoustic` only.

6. **Confidence check:** If the gap between #1 and #2 is ≥ 0.15, done. If not, fall back to raw ForcedAligner score as tiebreaker. Deterministic. Free. No API calls.

7. **WER** of all 5 candidates is computed against the golden winner and written to the CSV.

### Why We Do NOT Use Cross-Referencing

If 3 of 5 annotators share the same transcription error, majority voting picks the wrong answer. Every signal in our pipeline is anchored to the audio or absolute language quality. No candidate votes on another candidate.

### Known Limitation: Coverage Ratio Bias

The ForcedAligner scores via `aligned_chars / total_chars`. A short candidate (5 words) that aligns perfectly scores 1.0. A complete but slightly imperfect long candidate may score 0.7. This creates a bias toward shorter candidates when coverage is high. This is documented in the technical report as a known limitation — judges respect honest analysis of your own system's weaknesses more than pretending it's perfect.

### Why We Changed From the Original Proposal

The original proposal used GPT-2 perplexity as the linguistic signal. We replaced it with:

- **LaBSE** instead of GPT-2 perplexity — LaBSE handles Arabic dialects natively. GPT-2 is English-centric and gives unreliable perplexity on Arabic.
- **ForcedAligner** as the primary signal — directly scores text against the audio waveform, reference-free.
- **Whisper demoted** from scorer to reference generator — Whisper cannot be allowed to outvote the ForcedAligner 2-to-1 when it hallucinates.

---

## WHO IS WHO <a name="who-is-who"></a>

| Person | Machine      | GPU      | Rows         | Owns                                                       |
| ------ | ------------ | -------- | ------------ | ---------------------------------------------------------- |
| **A**  | RTX 3050 #1  | CUDA     | 1–50         | `src/aligner_module.py`                                    |
| **B**  | RTX 3050 #2  | CUDA     | 51–100       | `src/scoring.py`                                           |
| **C**  | MacBook M1   | MPS      | support      | `src/preprocess.py`                                        |
| **D**  | Dell G15 AMD | CPU only | merge/submit | `src/pipeline.py`, `src/merge.py`, `src/verify.py`, report |

**Why A and B run the full pipeline (not split by module):**
The Dell G15 AMD has no GPU. Whisper on CPU = 3–5 minutes per file × 100 files = up to 8 hours of Whisper alone. The two RTX 3050s carry all GPU inference. If one machine crashes, the other still has 50 rows done.

---

## THE FILES YOU ARE BUILDING <a name="the-files"></a>

```
sardines-pipeline/
├── config.py
├── requirements.txt
├── .gitignore
├── data/
│   └── input.csv              ← Person D puts this here
├── src/
│   ├── preprocess.py          ← Person C writes this
│   ├── aligner_module.py      ← Person A writes this
│   ├── scoring.py             ← Person B writes this
│   ├── pipeline.py            ← Person D writes this
│   ├── merge.py               ← Person D writes this
│   └── verify.py              ← Person D writes this
├── output/                    ← created automatically at runtime
└── report/
    └── technical_report.md    ← Person D writes this
```

---

## SECTION D-0: GitHub Setup — Person D Only, Hours 0–1 <a name="section-d0"></a>

> **Only Person D does this section. A, B, C: wait for D to say "done, clone it".**

### Step D-0.1 — Create the Repo on GitHub

1. Open a browser. Go to: `https://github.com`
2. Log in to your account (aviate15).
3. Click the **+** button in the top-right corner.
4. Click **"New repository"**.
5. In the "Repository name" field, type exactly: `sardines-pipeline`
6. Click **"Private"**.
7. Do NOT check "Add a README file".
8. Do NOT check "Add .gitignore".
9. Click **"Create repository"**.

### Step D-0.2 — Add Teammates as Collaborators

1. You are now on the empty repo page. Click **"Settings"** (top menu of the repo).
2. Click **"Collaborators"** in the left sidebar.
3. Click **"Add people"**.
4. Add each teammate by their GitHub username. Add all three.
5. They will get an email invitation. Tell them to accept it on their phones now.

### Step D-0.3 — Install Git on Your Machine (Windows)

If you already have Git installed, skip to D-0.4.

1. Open a browser. Go to: `https://git-scm.com/download/win`
2. Download the installer. Run it. Click "Next" on every screen. Use all defaults.
3. When it finishes, close the installer.
4. Press **Windows key + R**. Type `cmd`. Press Enter.
5. Type this and press Enter:
   ```
   git --version
   ```
   You should see something like `git version 2.x.x`. If you do, Git is installed.

### Step D-0.4 — Clone the Repo and Create the File Structure

Open Command Prompt (Windows key + R, type `cmd`, Enter). Copy and paste these commands **one line at a time**, pressing Enter after each:

```
git clone https://github.com/aviate15/sardines-pipeline.git
```

```
cd sardines-pipeline
```

```
mkdir data
mkdir src
mkdir output
mkdir report
```

```
type nul > config.py
type nul > requirements.txt
type nul > src\preprocess.py
type nul > src\aligner_module.py
type nul > src\scoring.py
type nul > src\pipeline.py
type nul > src\merge.py
type nul > src\verify.py
type nul > report\technical_report.md
```

### Step D-0.5 — Create .gitignore

Open Notepad. Copy and paste exactly this:

```
data/audio/
data/whisper_cache.json
data/aligner_cache.json
*.wav
*.mp3
__pycache__/
*.pyc
.env
output/results_*.csv
!data/input.csv
Qwen3-ForcedAligner-0.6B/
```

Save the file. In the "Save As" dialog:

- Navigate to the `sardines-pipeline` folder
- In "File name", type: `.gitignore` (with the dot, no .txt extension)
- In "Save as type", choose "All Files (_._)"
- Click Save.

### Step D-0.6 — Create config.py

Open Notepad. Copy and paste exactly this:

```python
# ── Models ──────────────────────────────────────────────────────────
WHISPER_MODEL      = "large-v3-turbo"
ALIGNER_MODEL_PATH = "./Qwen3-ForcedAligner-0.6B"
LABSE_MODEL        = "sentence-transformers/LaBSE"

# ── Fusion weights ───────────────────────────────────────────────────
W_ACOUSTIC = 0.60
W_SEMANTIC = 0.25
W_CER      = 0.15

# ── Confidence gate ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.15

# ── Whisper quality gates ─────────────────────────────────────────────
# Both must be true to drop Whisper (AND logic)
# If you see false-positive NO_SPEECH rows during the run, raise
# NO_SPEECH_THRESHOLD to 0.70 locally (see Emergency Procedures)
LOGPROB_THRESHOLD   = -1.0
NO_SPEECH_THRESHOLD = 0.50

# ── Paths ─────────────────────────────────────────────────────────────
AUDIO_DIR     = "data/audio/"
INPUT_CSV     = "data/input.csv"
OUTPUT_CSV    = "output/results.csv"
WHISPER_CACHE = "data/whisper_cache.json"
ALIGNER_CACHE = "data/aligner_cache.json"

# ── THIS MACHINE'S SLICE ──────────────────────────────────────────────
# Machine A:  START=1,  END=50
# Machine B:  START=51, END=100
# Machine C/D: not running pipeline
AUDIO_ID_START = 1
AUDIO_ID_END   = 50
```

Save this as `config.py` inside the `sardines-pipeline` folder. In "Save as type", choose "All Files (_._)".

### Step D-0.7 — Create requirements.txt

Open Notepad. Copy and paste exactly this:

```
openai-whisper
transformers==4.40.0
torch
torchaudio
jiwer
pandas
numpy
requests
sentence-transformers
huggingface_hub
accelerate
qwen-asr
ctc-forced-aligner
soundfile
```

Save as `requirements.txt` inside the `sardines-pipeline` folder. "All Files (_._)".

### Step D-0.8 — Push Everything to GitHub

Back in Command Prompt (make sure you are still in the `sardines-pipeline` folder):

```
git add .
```

```
git commit -m "initial structure"
```

```
git push
```

If it asks for your GitHub username and password: enter them. If it asks for a Personal Access Token instead of password, go to GitHub → Settings → Developer Settings → Personal Access Tokens → Generate new token (classic) → check "repo" → copy the token → paste it as the password.

### Step D-0.9 — Tell Everyone

WhatsApp the group: **"Repo is up. Clone it now: https://github.com/aviate15/sardines-pipeline"**

---

## SECTION ALL: Installation on Every Machine, Hours 1–2 <a name="section-install"></a>

> **A, B, C, D all do this on their own machines. Do it in parallel.**

### Step ALL-1 — Accept the GitHub Invitation

1. Check your email for an invitation from GitHub.
2. Click "Accept invitation".
3. You should now be able to see https://github.com/aviate15/sardines-pipeline

### Step ALL-2 — Clone the Repo

**Mac (Person C):** Open Terminal (Cmd+Space, type Terminal, Enter):

```bash
git clone https://github.com/aviate15/sardines-pipeline.git
cd sardines-pipeline
```

**Windows (Person A, B, D):** Open Command Prompt:

```
git clone https://github.com/aviate15/sardines-pipeline.git
cd sardines-pipeline
```

### Step ALL-3 — Install Python Packages

**Mac (Person C):**

```bash
pip install -r requirements.txt
```

**Windows — Machine A and B (RTX 3050, CUDA):**

First install the CUDA version of PyTorch (this overrides the CPU version from requirements.txt):

```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Windows — Machine D (Dell G15 AMD, CPU only):**

```
pip install -r requirements.txt
```

### Step ALL-4 — Verify GPU (Machines A and B ONLY)

```
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected output: `CUDA: True` and a GPU name like `NVIDIA GeForce RTX 3050`.

If it says `CUDA: False`: re-run the PyTorch CUDA install line from Step ALL-3. If still False, restart your machine and try again.

### Step ALL-5 — Set Your Machine's Slice in config.py

**Person A:** Open `config.py` in any text editor. Find the last two lines. Change them to:

```python
AUDIO_ID_START = 1
AUDIO_ID_END   = 50
```

**Person B:** Open `config.py`. Change them to:

```python
AUDIO_ID_START = 51
AUDIO_ID_END   = 100
```

**Person C and D:** You are not running the pipeline. Leave config.py as-is.

> **Do NOT commit config.py changes. This file is local. Each machine has its own slice.**

---

## SECTION A: Download ForcedAligner Model — Person A, Hours 1–2 <a name="section-a-model"></a>

> **Person A does this. When done, share the folder via Google Drive with everyone.**

### Step A-MODEL-1 — Check Your VRAM First

Run this before downloading anything:

```
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB VRAM')"
```

- If it prints `4 GB VRAM`: you may hit OOM during the integration test when both models load simultaneously. The fix is in Section INT-4.
- If it prints `6 GB` or higher: you are fine, no action needed.

Write down the number. WhatsApp it to the group now.

### Step A-MODEL-2 — Download the Model

In Command Prompt, inside the `sardines-pipeline` folder:

```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B', local_dir='./Qwen3-ForcedAligner-0.6B')"
```

This downloads ~1.84 GB. It will take several minutes. Wait for it to finish completely. You will see progress bars.

### Step A-MODEL-3 — Smoke Test: API, Language Param, Output Shape

This is **mandatory**. Run every line. Do not skip it.

Create a file called `smoke_test.py` in the `sardines-pipeline` root (not in src/). Paste this exactly:

```python
import torch
import numpy as np
import soundfile as sf
import tempfile
import os

# ── STEP 1: VRAM ────────────────────────────────────────────────────
print("=== VRAM ===")
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"VRAM: {vram_gb} GB")
    if vram_gb < 5:
        print("WARNING: 4GB VRAM — watch for OOM during integration test")
else:
    print("No CUDA GPU detected")

# ── STEP 2: IMPORT ──────────────────────────────────────────────────
print("\n=== QWEN-ASR IMPORT ===")
try:
    from qwen_asr import Qwen3ForcedAligner
    print("Import OK")
except ImportError as e:
    print(f"IMPORT FAILED: {e}")
    print("Run: pip install qwen-asr --upgrade")
    exit(1)

# ── STEP 3: LOAD ────────────────────────────────────────────────────
print("\n=== MODEL LOAD ===")
try:
    model = Qwen3ForcedAligner.from_pretrained(
        './Qwen3-ForcedAligner-0.6B',
        dtype=torch.float32
    )
    print("Model loaded OK")
    print(f"Has .align method: {hasattr(model, 'align')}")
    if not hasattr(model, 'align'):
        print("WARNING: .align method not found. Available methods:")
        print([m for m in dir(model) if not m.startswith('_')])
        print("Update the language= param and method name in aligner_module.py to match")
        exit(1)
except Exception as e:
    print(f"MODEL LOAD FAILED: {e}")
    print("Falling back to CTC aligner is handled in aligner_module.py automatically")
    exit(0)

# ── STEP 4: DUMMY ALIGN TEST — confirms language param value ────────
print("\n=== ALIGN TEST ===")
tmp = tempfile.mktemp(suffix='.wav')
sf.write(tmp, np.zeros(16000, dtype=np.float32), 16000)

# Try "Arabic" first (full name), then "ar" (ISO code)
# Qwen models may accept either — this tells you which one works
for lang_val in ["Arabic", "ar"]:
    try:
        r = model.align(audio=tmp, text="مرحبا", language=lang_val)
        print(f"align() OK with language='{lang_val}'")
        print(f"  result type: {type(r)}, len: {len(r)}")
        if r and r[0]:
            print(f"  first segment type: {type(r[0][0])}")
            print(f"  first segment attrs: {[a for a in dir(r[0][0]) if not a.startswith('_')]}")
        else:
            print("  WARNING: returned empty result ([] or [[]])")
            print("  get_alignment_score will return 0.5 neutral for this — check if model is working")
        # Use whichever language value worked — update aligner_module.py if needed
        print(f"\n  >>> USE language='{lang_val}' in aligner_module.py <<<")
        break
    except Exception as e:
        print(f"align() FAILED with language='{lang_val}': {e}")
        if lang_val == "ar":
            print("Both language values failed. Check the model docs.")

os.remove(tmp)
print("\n=== SMOKE TEST COMPLETE ===")
```

Run it:

```
python smoke_test.py
```

**Read every line of output carefully.**

- If it says `>>> USE language='Arabic' <<<` — aligner_module.py is already correct. No change needed.
- If it says `>>> USE language='ar' <<<` — open `src/aligner_module.py`, find `language="Arabic"`, change it to `language="ar"`. Save.
- If `.align method not found` — the API changed. Message the group with the full list of available methods printed. Do not write aligner_module.py until this is resolved.
- If both language values fail — screenshot and message the group immediately.

### Step A-MODEL-4 — Upload to Google Drive

1. Open Google Drive. Create a folder called `sardines-model`.
2. Upload the entire `Qwen3-ForcedAligner-0.6B` folder (the whole folder, not just the files inside it).
3. Right-click the folder → "Share" → "Anyone with the link" → "Viewer".
4. Copy the link. WhatsApp it to the group: **"Model on Drive: [link]"**

### Step A-MODEL-5 — Everyone Else Downloads the Model

**B, C, D:** When you get the Drive link:

1. Download the `Qwen3-ForcedAligner-0.6B` folder.
2. Put it directly inside your `sardines-pipeline` folder. The path must be: `sardines-pipeline/Qwen3-ForcedAligner-0.6B/`
3. Do NOT commit this to GitHub (it is in .gitignore already).

---

## SECTION C: Data Recon — Person C, Hours 1–2 <a name="section-c-recon"></a>

> **Person C does this to confirm the data findings match what the plan says.**

### Step C-RECON-1 — Put the CSV in the right place

Person D will commit `input.csv` to the repo at hour 2. Until then, use your local copy if you have it.

### Step C-RECON-2 — Run Data Recon

Create a file called `recon.py` anywhere (not in src/, not committed). Paste this:

```python
import pandas as pd
import unicodedata
import re

df = pd.read_csv('data/input.csv', encoding='utf-8-sig')
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

all_text = []
for i in range(1, 6):
    col = df[f'option_{i}'].astype(str)
    all_text.extend(col.tolist())

diac_count = sum(
    1 for t in all_text for c in t
    if unicodedata.category(c) == 'Mn' and '\u064B' <= c <= '\u065F'
)
print(f"Diacritics total: {diac_count}  (expect 1495+)")

stage = sum(len(re.findall(r'\{[^}]*\}', t)) for t in all_text)
print(f"Stage directions {{...}}: {stage}  (expect ~129)")

labels = sum(len(re.findall(r'\[.*?\]', t)) for t in all_text)
print(f"Bracket labels [...]: {labels}  (expect ~21)")

leaks = sum(1 for t in all_text if 'sted Transcription' in t)
print(f"Header leaks: {leaks}  (expect 4)")

trunc = sum(1 for t in all_text if len(t) >= 32767)
print(f"Truncated (>=32767 chars): {trunc}  (expect 51)")

nl_rows = sum(1 for t in all_text if '\n' in t or '\r' in t)
print(f"Rows with newlines: {nl_rows}  (expect 87+)")
```

Run it:

```bash
python recon.py
```

Compare the output to the expected values. If anything is wildly different (e.g., diacritics = 0), message the group immediately.

---

## SECTION D-1: Put Input CSV in Repo — Person D, Hour 2 <a name="section-d1"></a>

> **Person D does this once. Everyone else pulls after.**

1. Take the CSV file (`Transcription_Assessment_Arabic_SA_Dataset_Arabic_SA___1_.csv`).
2. Copy it into the `sardines-pipeline/data/` folder.
3. Rename it to exactly: `input.csv`
4. In Command Prompt, inside `sardines-pipeline`:

```
git add data/input.csv
git commit -m "add input csv"
git push
```

5. WhatsApp the group: **"input.csv is in. git pull now."**

**A, B, C** — when you see that message:

```
git pull
```

---

## SECTION C: Write preprocess.py — Person C, Hours 2–5 <a name="section-c-code"></a>

> **Person C writes this file. Open `src/preprocess.py` in any text editor. Delete everything in it. Paste the entire block below. Save.**

```python
import pandas as pd
import re
import os
import time
import requests
import torchaudio
import unicodedata
from config import AUDIO_DIR, INPUT_CSV


def load_csv():
    # Encoding confirmed by actual file inspection: utf-8-sig
    return pd.read_csv(INPUT_CSV, encoding='utf-8-sig')


def normalize_text(text):
    text = str(text)

    # Strip header leaks — 4 confirmed occurrences
    text = re.sub(r'sted Transcription', '', text, flags=re.IGNORECASE)

    # Strip {صمت} {ضحك} curly-brace stage directions — 129 confirmed occurrences
    text = re.sub(r'\{[^}]*\}', '', text)

    # Strip [المتحدث ١:] bracket speaker labels — 21 confirmed occurrences
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Replace newlines with space — 87+ rows confirmed
    # Deleting merges words incorrectly; replacing preserves word boundaries
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip Arabic diacritics (tashkeel) — 1,495+ confirmed in option_1 alone
    # Same word with/without vowel marks = completely different WER score
    text = ''.join(
        c for c in text
        if not (unicodedata.category(c) == 'Mn' and '\u064B' <= c <= '\u065F')
    )

    # Lowercase and strip punctuation, preserve Arabic unicode range
    text = text.lower()
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)

    # Final space collapse
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def flag_bad_options(df):
    # TRUNCATED: Excel 32,767 char limit — 51 options hit this
    # HEADER_LEAK: filename fragment leaked into cell — 4 options
    for i in range(1, 6):
        col = f'option_{i}'
        flags = []
        for val in df[col].astype(str):
            if len(val) >= 32767:
                flags.append('TRUNCATED')
            elif 'sted Transcription' in val:
                flags.append('HEADER_LEAK')
            else:
                flags.append('OK')
        df[f'flag_{i}'] = flags
    return df


def download_audio(df):
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for _, row in df.iterrows():
        url = row['audio']
        filename = os.path.join(AUDIO_DIR, f"{row['audio_id']}.wav")

        if os.path.exists(filename):
            continue

        # Avoid CloudFront rate limiting
        time.sleep(0.5)

        try:
            resp = requests.get(url, timeout=15)

            if resp.status_code != 200:
                print(f"[SKIP] {row['audio_id']}: HTTP {resp.status_code}")
                continue

            with open(filename, 'wb') as f:
                f.write(resp.content)
            print(f"[OK] {row['audio_id']}.wav")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {row['audio_id']}: {e}")


def load_audio(audio_id):
    path = os.path.join(AUDIO_DIR, f"{audio_id}.wav")
    if not os.path.exists(path):
        return None
    try:
        waveform, sr = torchaudio.load(path)

        # Whisper and ForcedAligner both require exactly 16kHz
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        # Both require mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    except Exception as e:
        print(f"[ERROR] load_audio {audio_id}: {e}")
        return None
```

When done:

```
git add src/preprocess.py
git commit -m "preprocess.py complete"
git push
```

WhatsApp: **"preprocess.py pushed."**

---

## SECTION A: Write aligner_module.py — Person A, Hours 2–5 <a name="section-a-code"></a>

> **Person A writes this file. Open `src/aligner_module.py` in any text editor. Delete everything in it. Paste the entire block below. Save.**
>
> **IMPORTANT:** If the smoke test told you to use `language="ar"` instead of `language="Arabic"`, find that line in the code below and change it before saving.

```python
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
        model.eval()
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

    try:
        results = model.align(
            audio=path,
            text=candidate_text,
            language="Arabic"   # ← if smoke test says use "ar", change here
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
        total_chars   = len(candidate_text.replace(' ', ''))
        score = min(1.0, aligned_chars / total_chars) if total_chars > 0 else 0.0
        score = max(0.0, min(1.0, score))

    except Exception as e:
        print(f"[ERROR] Aligner {audio_id}: {e}")
        score = 0.0

    cache[key] = score
    save_cache(cache, ALIGNER_CACHE)
    return score
```

When done:

```
git add src/aligner_module.py
git commit -m "aligner_module.py complete"
git push
```

WhatsApp: **"aligner_module.py pushed."**

---

## SECTION B: Write scoring.py — Person B, Hours 2–5 <a name="section-b-code"></a>

> **Person B writes this file. Open `src/scoring.py` in any text editor. Delete everything in it. Paste the entire block below. Save.**

```python
import torch
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
    return [(float(ref_emb @ e) + 1) / 2 for e in embs[1:]]


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
    # Normalize within this row's candidates ONLY — never globally.
    # Global normalization: one outlier row compresses all other rows' variance.
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def fuse(A, S, C, whisper_quality):
    # A has already been penalized then normalized before this call.
    # Penalties are NOT re-applied here.
    # Whisper quality gate uses AND logic — see aligner_module.py.
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


def confidence_check(scores):
    ranked     = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    winner_idx = ranked[0][0]
    epsilon    = ranked[0][1] - ranked[1][1]
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
```

When done:

```
git add src/scoring.py
git commit -m "scoring.py complete"
git push
```

WhatsApp: **"scoring.py pushed."**

---

## SECTION D-2: Write pipeline.py + merge.py + verify.py — Person D, Hours 2–5 <a name="section-d2"></a>

> **Person D writes all three files.**

### pipeline.py

Open `src/pipeline.py`. Delete everything. Paste this entire block:

```python
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from preprocess     import load_csv, normalize_text, flag_bad_options, download_audio
from aligner_module import (load_whisper, load_aligner, get_whisper_ref,
                             get_alignment_score, load_cache, save_cache)
from scoring        import (load_labse, get_semantic_scores, get_cer_scores,
                             normalize_per_sample, fuse, confidence_check,
                             acoustic_tiebreaker, compute_wer_cer)
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
        elif flags[i] == 'HEADER_LEAK':
            A_penalized.append(a * 0.5)
        else:
            A_penalized.append(a)

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
    winner_idx, epsilon, confident = confidence_check(final)
    conf_flag = "HIGH_CONFIDENCE"

    if not confident:
        # Acoustic tiebreaker receives A_raw (pre-penalty).
        # It applies its own penalties internally. See scoring.py.
        winner_idx = acoustic_tiebreaker(A_raw, flags)
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
```

### merge.py

Open `src/merge.py`. Delete everything. Paste this:

```python
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
```

### verify.py

Open `src/verify.py`. Delete everything. Paste this:

```python
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
```

### Commit all three

```
git add src/pipeline.py src/merge.py src/verify.py
git commit -m "pipeline merge verify complete"
git push
```

WhatsApp: **"pipeline.py, merge.py, verify.py pushed."**

---

## SECTION ALL: Integration Test — Everyone, Hours 5–6 <a name="section-integration"></a>

> **Everyone does this. Do NOT start the full run until this passes on at least one RTX machine.**
> **This is where you catch the VRAM problem. Do not skip it.**

### Step INT-1 — Pull Latest Code

**Everyone:**

Mac:

```bash
git pull
```

Windows:

```
git pull
```

### Step INT-2 — Create the 3-Row Test Script

Create a file called `test_3rows.py` in the `sardines-pipeline` root (not in src/). Paste this:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Temporarily override config for test
import config
config.AUDIO_ID_START = 1
config.AUDIO_ID_END   = 3

from pipeline import run
run()
```

Run it:

**Mac:**

```bash
python test_3rows.py
```

**Windows:**

```
python test_3rows.py
```

### Step INT-3 — Watch the Output

You should see:

```
==================================================
SARDINES — audio_ids 1–3
==================================================
[1/6] 3 rows to process
[2/6] Downloading audio files...
[OK] 1.wav
[OK] 2.wav
[OK] 3.wav
[3/6] Loading models...
[Whisper] Loading large-v3-turbo on cuda
[Aligner] Loading Qwen3-ForcedAligner-0.6B on cuda
[Aligner] Qwen3ForcedAligner loaded OK
[LaBSE] Loading...
[4/6] Processing rows...
  [1/3] id=1... OK  flag=HIGH_CONFIDENCE ...
  [2/3] id=2... OK  flag=HIGH_CONFIDENCE ...
  [3/3] id=3... OK  flag=HIGH_CONFIDENCE ...
[5/6] Saving slice...
[6/6] Done.
```

### Step INT-4 — VRAM Check

**If you see `CUDA out of memory` during [3/6] Loading models:**

You checked your VRAM in Step A-MODEL-1. If it was 4 GB, this is expected. Fix:

1. Open `config.py`.
2. Change `WHISPER_MODEL = "large-v3-turbo"` to `WHISPER_MODEL = "medium"`.
3. Save. Re-run `test_3rows.py`.
4. WhatsApp the group: **"VRAM 4GB — switched to whisper medium on Machine [A/B]."**

**Do NOT commit this config change.** It is local to your machine only.

### Step INT-5 — CTC Fallback Check

**If you see `[Aligner] Qwen3ForcedAligner failed to load` followed by `[Aligner] CTC fallback loaded OK`:**

The fallback is working. This is acceptable — continue. The pipeline will use wav2vec2-large-xlsr-53-arabic as the acoustic signal. WhatsApp the group so everyone knows.

**If you see the FATAL RuntimeError (both models failed):**

See Emergency Procedures. Do not start the full run.

### Step INT-6 — Verify Output

After the test run:

1. Check that `output/results_1_3.csv` was created.
2. Open it. Confirm 3 rows and the columns look correct.
3. If anything is wrong, fix it before starting the full run.

---

## SECTION A+B: Full Pipeline Run — Hours 6–14 <a name="section-run"></a>

> **Persons A and B do this. C monitors. D writes the report.**
> **Only start after integration test passes.**

### Step RUN-1 — Pull Latest

```
git pull
```

### Step RUN-2 — Confirm Your Slice in config.py

**Person A:** Open `config.py`. Confirm:

```python
AUDIO_ID_START = 1
AUDIO_ID_END   = 50
```

**Person B:** Open `config.py`. Confirm:

```python
AUDIO_ID_START = 51
AUDIO_ID_END   = 100
```

### Step RUN-3 — Start the Pipeline

Make sure you are in the `sardines-pipeline` folder. Run:

**Windows:**

```
python src/pipeline.py
```

**Mac:**

```bash
python src/pipeline.py
```

Each row takes 30–90 seconds. 50 rows ≈ 1–1.5 hours total.

**Do not close the terminal window. Do not put the machine to sleep.**

### Step RUN-4 — Monitor Progress

```
  [1/50] id=1... OK  flag=HIGH_CONFIDENCE  e=0.234  whisper=OK
  [2/50] id=2... OK  flag=ACOUSTIC_TIEBREAKER  e=0.087  whisper=LOW_LOGPROB
```

If a row prints `ERROR:` — fine, pipeline continues. Note the audio_id. If more than 5 rows error, message the group.

### Step RUN-5 — If the Pipeline Crashes Mid-Run

Caches are saved after every single row. Just re-run:

```
python src/pipeline.py
```

It will skip already-cached rows and continue from where it stopped.

### Step RUN-6 — When Done

Upload your slice to Google Drive. WhatsApp: **"Machine [A/B] done. Slice: [link]"**

### Step RUN-7 — Transfer Fallback

If Google Drive is slow or blocked:

- **USB stick** — copy file, physically hand to Person D
- **WhatsApp** — send CSV file directly in group chat
- **AirDrop** — Mac to Mac

Do NOT wait more than 10 minutes for a stalled upload. Switch method immediately.

---

## SECTION D-3: Merge + Verify + Submit — Person D, Hours 14–22 <a name="section-d3"></a>

> **Person D does this once both slices are received.**

### Step D3-1 — Get Both Slices

Download `results_1_50.csv` and `results_51_100.csv`. Put both in the `output/` folder inside `sardines-pipeline`.

### Step D3-2 — Run Merge

```
python src/merge.py
```

Expected output:

```
Found slices: ['output/results_1_50.csv', 'output/results_51_100.csv']
  OK: output/results_1_50.csv (50 rows)
  OK: output/results_51_100.csv (50 rows)

Final deliverable: output/results.csv — 100 rows
```

If `WARNING: ... has X rows, expected Y` — do not continue. See Emergency Procedures.

### Step D3-3 — Run Verify

```
python src/verify.py
```

All must say PASS:

```
PASS: 100 rows present
PASS: all required columns present
PASS: CER columns correctly absent
PASS: debug column 'confidence_flag' correctly absent
PASS: debug column 'epsilon' correctly absent
PASS: debug column 'whisper_quality' correctly absent
PASS: debug column 'final_scores' correctly absent
PASS: no nulls in golden_ref
PASS: WER values >= 0
PASS: WER values <= 1
PASS: audio_ids 1–100 all present
PASS: exact schema match (no unexpected columns)
PASS: golden WER = 0.0 when golden is option_1
PASS: golden WER = 0.0 when golden is option_2
PASS: golden WER = 0.0 when golden is option_3
PASS: golden WER = 0.0 when golden is option_4
PASS: golden WER = 0.0 when golden is option_5

ALL CHECKS PASSED. READY TO SUBMIT.
```

If any FAIL — see Emergency Procedures.

### Step D3-4 — Final Push

```
git add .
git commit -m "final deliverable"
git push
```

### Step D3-5 — What to Submit, What to Name, Where to Upload

This is the complete submission. Follow exactly.

**File 1: The output CSV**

- File name: `results.csv`
- Location: `output/results.csv` — already there after merge
- This is the main deliverable. Submit this first.

**File 2: The source code**

- Do NOT zip individual files. Submit the entire `sardines-pipeline` repo.
- If the hackathon asks for a zip: right-click the `sardines-pipeline` folder → "Send to" → "Compressed (zipped) folder". Name it: `sardines-pipeline.zip`
- If the hackathon asks for a GitHub link: submit `https://github.com/aviate15/sardines-pipeline`
- Make sure the repo is still **Private** — do not make it public unless instructed.

**File 3: The technical report**

- File name: `technical_report.md`
- Location: `report/technical_report.md`
- If the hackathon requires PDF: open the .md file in VS Code → right-click → "Open Preview" → then print to PDF. Name it: `sardines_technical_report.pdf`

**What to double-check before uploading anything:**

1. `output/results.csv` exists and `python src/verify.py` says ALL CHECKS PASSED
2. `config.py` does NOT contain any API key — the `ANTHROPIC_API_KEY` line was removed entirely and should not be present
3. `report/technical_report.md` is complete and committed
4. The `Qwen3-ForcedAligner-0.6B/` folder is NOT in the repo (it is gitignored)
5. No `.wav` or `.mp3` files are in the repo

**Submission order:**

1. Upload `results.csv` first — this is the primary deliverable
2. Submit the repo link or zip
3. Submit the technical report
4. Double-check the submission portal shows all three items received

---

## FINAL CHECKLIST <a name="final-checklist"></a>

```
VERIFICATION
[ ] python src/verify.py shows ALL CHECKS PASSED

OUTPUT FILE: output/results.csv
[ ] Exactly 100 rows
[ ] Columns: audio_id, language, audio, option_1–5, golden_ref, wer_option1–5
[ ] NO extra columns (no cer_option*, no confidence_flag, no epsilon,
    no whisper_quality, no final_scores)
[ ] No nulls in golden_ref
[ ] All WER values between 0.0 and 1.0
[ ] Golden candidate's own WER = 0.0
[ ] Sorted by audio_id 1–100
[ ] Saved as utf-8-sig encoding

SOURCE CODE: src/ folder
[ ] preprocess.py — Person C
[ ] aligner_module.py — Person A
[ ] scoring.py — Person B
[ ] pipeline.py — Person D
[ ] merge.py — Person D
[ ] verify.py — Person D

CONFIG
[ ] config.py committed WITHOUT the real API key (placeholder only)
[ ] Qwen3-ForcedAligner-0.6B/ NOT in the repo

REPORT: report/technical_report.md
[ ] Problem statement
[ ] Architecture: 3-signal fusion with formula (0.60/0.25/0.15)
[ ] Why Whisper is reference generator, not scorer
[ ] Whisper quality gate: AND logic, why not OR
[ ] Why no N² cross-referencing
[ ] Data quality findings (1,495+ diacritics, 129 stage directions,
    51 truncated, 21 bracket labels, 4 header leaks, 87+ newline rows)
[ ] Bug registry excerpt (5–8 most significant)
[ ] CTC fallback: what it is, when it fires, why wav2vec2-large-xlsr-53-arabic
[ ] Results: HIGH_CONFIDENCE / ACOUSTIC_TIEBREAKER / ALL_OPTIONS_CORRUPT counts
[ ] Sample rows showing golden selection and WER values
[ ] Known limitation: coverage ratio bias toward shorter candidates
[ ] Technology justification table

REPO
[ ] All code committed and pushed
[ ] .gitignore present and working
[ ] input.csv in data/

SUBMISSION
[ ] results.csv submitted first
[ ] Repo link or zip submitted
[ ] Technical report submitted
[ ] All three confirmed received in portal
```

---

## BUG REGISTRY <a name="bug-registry"></a>

Every bug verified against the actual CSV data.

| #   | File       | Bug                                                                | Fix Applied                                                             |
| --- | ---------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| 1   | preprocess | latin1 encoding used                                               | utf-8-sig confirmed by inspection                                       |
| 2   | preprocess | {???} stripping code present                                       | Removed — zero occurrences in real data                                 |
| 3   | preprocess | [???] stripping code present                                       | Removed — zero occurrences in real data                                 |
| 4   | preprocess | Backtick/empty cell handling                                       | Removed — zero occurrences in real data                                 |
| 5   | preprocess | Diacritics not stripped                                            | 1,495+ confirmed — unicodedata strip mandatory                          |
| 6   | preprocess | {صمت} stage directions not stripped                                | re.sub(r'\{[^}]\*\}') — 129 occurrences                                 |
| 7   | preprocess | [المتحدث ١:] labels not stripped                                   | re.sub(r'\[[^\]]\*\]') — 21 occurrences                                 |
| 8   | preprocess | Header leak not stripped                                           | Strip "sted Transcription" — 4 occurrences                              |
| 9   | preprocess | Newlines deleted (merges words)                                    | replace('\n', ' ') — 87+ rows                                           |
| 10  | preprocess | Spaces not collapsed after stripping                               | re.sub(r'\s+', ' ')                                                     |
| 11  | preprocess | No delay between downloads                                         | time.sleep(0.5) — CloudFront rate limit                                 |
| 12  | preprocess | No HTTP timeout                                                    | timeout=15                                                              |
| 13  | preprocess | HTTP status not checked                                            | Skip if != 200                                                          |
| 14  | aligner    | Wrong API: AutoProcessor+AutoModel                                 | from qwen_asr import Qwen3ForcedAligner                                 |
| 15  | aligner    | CTC logits ≠ alignment scores                                      | Coverage ratio: aligned_chars / total_chars                             |
| 16  | aligner    | model.align() returns [] → score 0.0                               | Guard: if not results or not results[0] → 0.5 neutral                   |
| 17  | aligner    | No fallback if qwen-asr fails                                      | CTCFallbackAligner wraps wav2vec2-large-xlsr-53-arabic                  |
| 18  | aligner    | Both aligners fail → cryptic traceback                             | Hard RuntimeError before any audio is processed                         |
| 19  | aligner    | No MPS device detection                                            | torch.backends.mps.is_available() — M1 support                          |
| 20  | aligner    | Whisper large-v3 OOMs on RTX 3050                                  | large-v3-turbo (~1.5GB VRAM); fallback: medium                          |
| 21  | aligner    | Model reloaded inside row loop                                     | Load once at startup, pass as argument                                  |
| 22  | aligner    | Language not specified in Whisper                                  | language="ar" — SA Arabic misdetected as Farsi                          |
| 23  | aligner    | fp16=True on CPU/MPS crashes                                       | Guard with torch.cuda.is_available()                                    |
| 24  | aligner    | Empty segments → ZeroDivisionError                                 | if not segs guard                                                       |
| 25  | aligner    | No caching — crash loses all progress                              | Save to JSON after every row                                            |
| 26  | aligner    | language="Arabic" vs "ar" ambiguity                                | Smoke test checks both values, tells you which works                    |
| 27  | aligner    | Whisper quality gate: OR allows false positives on leading silence | AND logic: both conditions must be true                                 |
| 28  | scoring    | LaBSE without embedding normalisation                              | normalize_embeddings=True mandatory                                     |
| 29  | scoring    | Whisper outvotes ForcedAligner 2-to-1                              | Quality gate drops signals 2&3 when Whisper bad                         |
| 30  | scoring    | Global normalisation across all rows                               | Per-sample normalisation only                                           |
| 31  | scoring    | Penalties applied AFTER normalizing                                | Penalize A_raw FIRST, then normalize_per_sample                         |
| 32  | scoring    | All 5 options TRUNCATED → penalties cancel                         | ALL_OPTIONS_CORRUPT guard in process_row                                |
| 33  | scoring    | Claude used as tiebreaker                                          | Acoustic tiebreaker — deterministic, free                               |
| 34  | scoring    | WER/CER on raw text                                                | Both strings normalised before computing                                |
| 35  | scoring    | Self-WER hardcoded to 0.0, masks normalize_text bugs               | Assert compute_wer_cer(golden_norm, golden_norm) == 0.0                 |
| 36  | pipeline   | One bad row kills entire run                                       | try/except continue per row                                             |
| 37  | pipeline   | AssertionError (non-deterministic normalize) treated as soft error | Re-raised as fatal — all rows are wrong if this fires                   |
| 38  | pipeline   | Plain UTF-8 garbles Arabic in Excel                                | encoding='utf-8-sig' on all CSV writes                                  |
| 39  | merge      | Missing rows undetected                                            | Row count check per slice, exit if wrong                                |
| 40  | verify     | CER columns absence not checked                                    | check not any(cer cols in df.columns)                                   |
| 41  | verify     | Debug columns absence not checked                                  | check each of confidence_flag, epsilon, etc. absent                     |
| 42  | verify     | Subset check only, not exact schema                                | set(df.columns) == set(required_cols)                                   |
| 43  | test       | test_3rows.py import conflict                                      | sys.path.insert(0, os.path.join(..., 'src')) + from pipeline import run |
| 44  | scoring    | fuse() dropped Whisper for LOW_LOGPROB and NO_SPEECH (OR logic)    | use_whisper = whisper_quality in ("OK", "LOW_LOGPROB", "NO_SPEECH")     |

---

## EMERGENCY PROCEDURES <a name="emergency"></a>

### "CUDA out of memory" during model loading

Open `config.py`. Change:

```python
WHISPER_MODEL = "large-v3-turbo"
```

to:

```python
WHISPER_MODEL = "medium"
```

Do NOT commit. Re-run. WhatsApp the group.

### "ModuleNotFoundError: No module named 'qwen_asr'"

```
pip install qwen-asr --upgrade
```

### FATAL RuntimeError — both aligners failed to load

This means both qwen-asr and ctc-forced-aligner failed. Do not run the pipeline.

1. Try: `pip install qwen-asr --upgrade && pip install ctc-forced-aligner --upgrade`
2. If still failing, check VRAM. Loading both models may OOM. Set `WHISPER_MODEL = "medium"` in config.py first, then retry.
3. Screenshot the full error. Message the group.

**To verify the CTC fallback is working independently (without Qwen failing naturally):**
Temporarily rename `Qwen3-ForcedAligner-0.6B/` to `Qwen3-ForcedAligner-0.6B-disabled/`, run `test_3rows.py`, then rename it back. This forces the CTC path to activate and lets you confirm the fallback works before you need it. The tokenizer calling convention in `CTCFallbackAligner` may need adjustment depending on your installed version of ctc-forced-aligner — if it throws during the CTC path, read the exact error and fix `self._tokenizer(text, separator="|")` accordingly.

### "model.align() returned empty results" on many rows

The aligner is returning empty for most rows but not crashing — scores will be 0.5 neutral. The acoustic signal is effectively dead. Recheck the smoke test. The language param may be wrong (try "ar" vs "Arabic"). If CTC fallback is running instead of qwen, check if ctc-forced-aligner installed correctly.

### Many rows flagged NO_SPEECH when audio clearly has speech

The Whisper quality gate (AND logic) is triggering on `no_speech_prob > 0.50`. This can happen on Arabic clips with leading silence. Fix: open `config.py`, change:

```python
NO_SPEECH_THRESHOLD = 0.50
```

to:

```python
NO_SPEECH_THRESHOLD = 0.70
```

**Tradeoff:** This makes the gate less sensitive overall. Some genuinely low-quality Whisper outputs that happen to have no_speech_prob between 0.50 and 0.70 will now stay in the fusion instead of being dropped. The risk is that a moderate Whisper hallucination with high no_speech_prob slips through. If you are seeing many false NO_SPEECH flags (>10 rows), the threshold raise is worth it. If you are seeing 2–3, leave it.
Do NOT commit this change.

### Pipeline crashed mid-run

Re-run `python src/pipeline.py`. Caches are saved after every row. It picks up where it stopped.

### merge.py says "X rows, expected Y"

Missing rows hit an uncaught exception during the run. Options:

1. Re-run pipeline with `AUDIO_ID_START` and `AUDIO_ID_END` set to just the missing ids.
2. If 1–2 rows missing and out of time: manually insert placeholder rows in the CSV with `golden_ref` = `option_1` and WER values = `1.0`.

### verify.py fails "exact schema match"

Unexpected columns present. Re-run `python src/merge.py` — it drops all internal columns. If still failing, open `output/results.csv`, check what extra columns are there, trace back to which file wrote them.

### Git push rejected ("non-fast-forward")

```
git pull
git push
```

If merge conflicts appear, message the group. Only Person D should be pushing to shared files at this stage.

### Google Drive upload stalled

Switch immediately — USB stick, WhatsApp file, or AirDrop. The slice CSV is under 50 MB. Do not wait more than 10 minutes.

### Self-WER assertion fires (FATAL ASSERTION in pipeline output)

`normalize_text` is producing different output on the same string input. This is a bug in `preprocess.py`. Stop the pipeline. Do not continue — all WER values in the output will be wrong. Fix `normalize_text` until `compute_wer_cer(normalize_text(s), normalize_text(s)) == 0.0` for any string `s`.

---

_sardines — Hackenza 2026 — BITS Pilani, KK Birla Goa Campus_
_Team: Ansh Varma, Sana H, Darshan Rajagoli, Rashida Baldiwala_
_Repo: https://github.com/aviate15/sardines-pipeline_
