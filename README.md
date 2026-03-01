# sardines — Automated Golden Transcription Selection Pipeline

**Hackenza 2026 · BITS Pilani, KK Birla Goa Campus**

**Team:** Ansh Varma (2024A7PS0614G) · Sana H (2024A7PS0561G) · Darshan Rajagoli (2024ADPS0685G) · Rashida Baldiwala (2024A7PS0631G)

**Client:** Renan Partners Private Limited

---

## Problem Statement

Renan Partners Private Limited provided a dataset of 100 Arabic (Saudi dialect) audio clips, each accompanied by five candidate transcriptions from different annotators or automated systems. The task was to build a **fully automated system** that:

1. Identifies the single most accurate transcription (the **golden reference**) for each clip.
2. Computes the Word Error Rate (WER) of all five transcriptions against the selected golden reference.
3. Outputs a clean CSV in the exact required format: `audio_id, language, audio, option_1–5, golden_ref, wer_option1–5`.

The system must run without human intervention and must be robust to differences in audio quality, transcription formatting, and Arabic dialect variation.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Total audio clips processed | 100 / 100 |
| Language | Arabic (Saudi dialect, `Arabic_SA`) |
| Whisper reference quality | 100% OK (0 low-confidence rows) |
| Mean fused score of selected golden option | **0.947** (range 0.832 – 0.997) |
| Mean confidence margin (ε) between top two options | 0.0154 |
| HIGH\_CONFIDENCE selections (ε ≥ 0.05) | 5 rows |
| ACOUSTIC\_TIEBREAKER selections (ε < 0.05) | 95 rows |
| Exact tied scores requiring raw acoustic tiebreak | 29 rows |
| Mean WER of rejected candidates vs. golden | 0.2475 |
| Self-consistency check (golden\_ref == option\_[predicted]) | **100 / 100 ✓** |
| Pipeline crashes / missing rows | **0** |

---

## Core Logic & Solution Approach

### Why Not Majority Voting?

The naive approach — picking whichever candidate appears most often across annotators — is unreliable. If three annotators share the same *systematic transcription error*, majority voting actively selects the wrong answer. Every signal in our pipeline is anchored independently to the raw audio waveform or to absolute linguistic quality. No candidate votes on another candidate.

### Three-Signal Fusion Pipeline

Our pipeline fuses three independent quality signals into a single confidence-ranked score per candidate:

**1. Forced Acoustic Alignment (weight: 0.40)**
We use a forced aligner to measure how well each candidate *fits* the audio at the character level. This gives a direct audio-text correspondence score that is **reference-free** — it requires no ground truth. This is the most principled possible acoustic signal.

**2. Multilingual Semantic Similarity via LaBSE (weight: 0.30)**
We use **LaBSE** (Language-agnostic BERT Sentence Embeddings), trained on 109 languages including Arabic, to score semantic similarity between each candidate and a Whisper-generated reference. This replaces GPT-2 perplexity, which is unreliable for Arabic since GPT-2 was trained predominantly on English.

**3. Character Error Rate vs. Whisper Reference (weight: 0.30)**
Whisper is used purely as a *reference generator*, not a scorer. Its output anchors two comparative signals (semantic similarity and CER). When Whisper is not confident, it is dropped entirely and the pipeline falls back to the acoustic alignment signal alone — preventing Whisper hallucinations from corrupting selection.

### Final Score Formula

```
fused_score = 0.40 × acoustic_score + 0.30 × semantic_score + 0.30 × (1 − CER)
```

The candidate with the highest fused score is selected as the golden reference. If the margin ε between the top two candidates is ≥ 0.05, it is flagged `HIGH_CONFIDENCE`. If ε < 0.05, a raw acoustic tiebreaker decides.

---

## System Architecture

```
Input CSV (100 rows × 5 candidates)
         │
         ▼
┌─────────────────────────────┐
│  preprocess.py              │
│  • Load CSV (utf-8-sig)     │
│  • Flag TRUNCATED / HEADER  │
│  • normalize_text()         │
│  • Download WAV files       │
└────────────┬────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────┐
│  pipeline.py  (per audio_id loop)                          │
│                                                            │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ ForcedAligner│  │ WhisperModule │  │ LaBSEScorer     │  │
│  │ acoustic     │  │ reference_gen │  │ semantic_sim    │  │
│  │ score ×5     │  │ (cached)      │  │ score ×5        │  │
│  └──────┬───────┘  └──────┬────────┘  └────────┬────────┘  │
│         │                 │                    │           │
│         └─────────────────┼────────────────────┘           │
│                           ▼                                │
│                    fuse.py → fused_score ×5                │
│                           │                                │
│                    selector.py → golden_ref                │
│                           │                                │
│                    wer_calc.py → wer_option1–5             │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
                   output/results.csv
```

---

## Setup & Installation

### Prerequisites

- Before running anything, read [ALL_PREREQUISITES.txt](./ALL_PREREQUISITES.txt) and make sure every item in it is installed on your machine.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aviate15/sardines-pipeline.git
cd sardines

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place input CSV at:
data/input.csv

# 4. Download Qwen3-ForcedAligner-0.6B into the project root:
# ./Qwen3-ForcedAligner-0.6B/
# (Available on HuggingFace: https://huggingface.co/qwen-asr/Qwen3-ForcedAligner-0.6B)
```

### Running the Pipeline

```bash
# 1. Configure machine slice in config.py (for single machine, leave defaults):
#    AUDIO_ID_START = 1
#    AUDIO_ID_END = 100

# 2. Run the main pipeline
python pipeline.py

# 3. (Optional) If running on two machines in parallel, merge slices:
python cross_merge.py

# 4. Verify output integrity
python verify.py
```

### Parallel Execution (Two Machines)

The pipeline supports splitting the workload across machines by configuring `AUDIO_ID_START` and `AUDIO_ID_END` in `config.py`:

- **Machine A:** `AUDIO_ID_START = 1`, `AUDIO_ID_END = 50`
- **Machine B:** `AUDIO_ID_START = 51`, `AUDIO_ID_END = 100`

After both machines finish, run `python cross_merge.py` to produce the final merged output.

---

## Project Structure

```
sardines/
├── config.py               # Global settings (paths, weights, thresholds)
├── pipeline.py             # Main entry point — orchestrates per-row processing
├── preprocess.py           # CSV loading, text normalization, WAV download
├── fuse.py                 # Weighted score fusion logic
├── selector.py             # Golden candidate selection + tiebreaker
├── wer_calc.py             # WER/CER computation (jiwer)
├── cross_merge.py          # Merges parallel machine outputs
├── verify.py               # Output validation and schema checks
├── requirements.txt        # Python dependencies
├── data/
│   ├── input.csv           # Input dataset (100 rows × 5 candidates)
│   ├── whisper_cache.json  # Cached Whisper reference transcripts
│   └── aligner_cache.json  # Cached forced alignment scores
├── output/
│   ├── results.csv         # Submission file (exact required schema)
│   └── results_enhanced.csv # Debug file with scores, confidence flags, epsilon
└── Qwen3-ForcedAligner-0.6B/  # Downloaded aligner model weights
```

---

## Technology Stack

| Component | Library / Model | Purpose |
|---|---|---|
| ForcedAligner (primary) | Qwen3-ForcedAligner-0.6B (`qwen-asr`) | Acoustic coverage scoring |
| ForcedAligner (fallback) | wav2vec2-large-xlsr-53-arabic (`ctc-forced-aligner`) | Acoustic fallback |
| Whisper | openai-whisper large-v3-turbo | Reference transcription + quality signal |
| Semantic similarity | sentence-transformers/LaBSE | Multilingual embedding similarity |
| WER / CER | jiwer | Character and word error rate |
| Audio loading | torchaudio | 16 kHz mono WAV preprocessing |
| Data handling | pandas | CSV I/O, utf-8-sig encoding |
| Deep learning | torch, transformers, accelerate | Model inference |

---

## Unit Testing & Verification

### Automated Verification (`verify.py`)

The `verify.py` script validates the final output against all submission requirements:

- Exact column schema check
- Audio ID completeness (1–100, no gaps)
- Golden WER = 0.0 for all rows (the golden reference scores perfectly against itself)
- No debug columns present in the submission file
- UTF-8 encoding compliance

Run it after pipeline completion:

```bash
python verify.py
```

A clean run produces: `✓ All 100 rows passed verification.`


### Self-Consistency Check

The pipeline includes an internal self-consistency check: `golden_ref == option_[predicted]`. This passed for all **100 / 100** rows, confirming that the selected golden transcription is always a verbatim match for one of the five input candidates (no hallucination or hybrid outputs).

### Unit Testing & Runtime Assertions

Beyond post-run verification, the pipeline includes strict, row-level unit tests executed at runtime. 
* **Self-WER Assertion:** Every processed row runs an end-to-end normalization test (`compute_wer_cer(golden_norm, golden_norm) == 0.0`). 
* This guarantees that the `normalize_text()` function is completely deterministic. 
* If this assertion fails, the pipeline halts immediately rather than silently writing incorrect WER values to the output.


### Additional Features: Caching & Crash Recovery
All model outputs are JSON-cached after every row. If the pipeline crashes at row 73, restarting it resumes automatically from row 74 with no data loss.

---
## Accessibility & Usability Considerations

While this is a backend pipeline without a graphical interface, we built it with strict data accessibility and global usability standards in mind:
* **Cross-Platform Data Accessibility (UTF-8 BOM):** All output CSVs are explicitly encoded with `utf-8-sig` (UTF-8 with BOM). This ensures that Arabic characters render perfectly when opened in Microsoft Excel on Windows, preventing the mojibake (garbled text) issues common in cross-platform data transfers.
* **Dialect-Inclusive NLP (LaBSE):** Standard language models often struggle with non-Western languages or specific dialects. We specifically selected LaBSE for our semantic scoring because it was trained on 109 languages and naturally handles the morphology and vocabulary of the Saudi Arabic dialect without structural bias.
* **Developer Accessibility:** If the primary forced aligner (Qwen3) fails due to hardware constraints (like VRAM exhaustion), the system automatically degrades gracefully to a fallback model (`wav2vec2-large-xlsr-53-arabic`). If both fail, it raises actionable error messages rather than failing silently, ensuring the pipeline is accessible to developers running varying hardware setups.

## Robustness & Error Handling

The pipeline handles multiple categories of data corruption:

- **TRUNCATED candidates** — detected and penalized during preprocessing
- **HEADER artifacts** — filtered before scoring
- **ALL_OPTIONS_CORRUPT** — hard guard for rows where all five options are corrupted simultaneously
- **Forced aligner neutral fallback** — returns 0.5 (neutral) instead of 0.0 when the aligner fails due to audio conditions, avoiding false penalization
- **Aligner model fallback chain** — automatically switches from Qwen3-ForcedAligner to wav2vec2-large-xlsr-53-arabic on VRAM exhaustion, with a hard `RuntimeError` if both fail

---

## Output Files

| File | Description |
|---|---|
| `output/results.csv` | **Submission file** — 100 rows, exact required schema |
| `output/results_enhanced.csv` | Debug file — includes scores, confidence flags, epsilon |
| `data/whisper_cache.json` | Whisper output cache (100 entries) |
| `data/aligner_cache.json` | Alignment score cache (up to 500 entries) |

---

*sardines — Hackenza 2026 — BITS Pilani, KK Birla Goa Campus*
*Ansh Varma · Sana H · Darshan Rajagoli · Rashida Baldiwala*
