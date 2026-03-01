# Technical Report: Automated Golden Transcription Selection Pipeline
### sardines, Hackenza 2026
### BITS Pilani, KK Birla Goa Campus

**Team:** Ansh Varma (2024A7PS0614G) · Sana H (2024A7PS0561G) · Darshan Rajagoli (2024ADPS0685G) · Rashida Baldiwala (2024A7PS0631G)

**Client:** Renan Partners Private Limited

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Why Our Approach is Superior](#3-why-our-approach-is-superior)
4. [System Architecture](#4-system-architecture)
5. [Data Preprocessing and Quality Analysis](#5-data-preprocessing-and-quality-analysis)
6. [Scoring Methodology](#6-scoring-methodology)
7. [Fusion, Confidence, and Selection Logic](#7-fusion-confidence-and-selection-logic)
8. [Implementation Details and Engineering Decisions](#8-implementation-details-and-engineering-decisions)
9. [Results and Analysis](#9-results-and-analysis)
10. [Novel Approach and Competitive Differentiation](#10-novel-approach-and-competitive-differentiation)
11. [Robustness Across Languages and Audio Conditions](#11-robustness-across-languages-and-audio-conditions)
12. [Known Limitations](#12-known-limitations)
13. [Future Improvements](#13-future-improvements)
14. [Reproducibility and Submission Checklist](#14-reproducibility-and-submission-checklist)

---

## 1. Executive Summary

sardines built a fully automated, multi-signal pipeline to identify the single most accurate (golden) transcription from five divergent candidates for 100 Arabic (Saudi dialect) audio clips. The pipeline fuses three independent quality signals, forced acoustic alignment, multilingual semantic similarity, and character error rate against a Whisper-generated reference, into a single confidence-ranked score per candidate.

**Key results at a glance:**

| Metric | Value |
|---|---|
| Total audio clips processed | 100 / 100 |
| Language | Arabic (Saudi dialect, `Arabic_SA`) |
| Whisper reference quality | 100% OK (0 low-confidence rows) |
| Mean fused score of selected golden option | **0.947** (range 0.832 – 0.997) |
| Mean confidence margin (ε) between top two options | 0.0154 |
| HIGH\_CONFIDENCE selections (ε ≥ 0.05) | **5 rows** |
| ACOUSTIC\_TIEBREAKER selections (ε < 0.05) | **95 rows** |
| Exact tied scores requiring raw acoustic tiebreak | **29 rows** |
| Mean WER of rejected candidates vs. golden | **0.2475** |
| Self-consistency check (golden\_ref == option\_[predicted]) | **100 / 100** ✓ |
| Pipeline crashes / missing rows | **0** |

All 100 rows were processed without error. The final deliverable `output/results.csv` passes all checks in `verify.py`, including exact column schema, audio\_id completeness (1–100), golden WER = 0.0, and no debug columns in the submission file.

---

## 2. Problem Statement

Renan Partners Private Limited provided a dataset of 100 Arabic audio clips, each accompanied by five candidate transcriptions from different annotators or automated systems. The task was to build a fully automated system that:

1. Identifies the single most accurate transcription (the **golden reference**) for each clip.
2. Computes the Word Error Rate (WER) of all five transcriptions against the selected golden reference.
3. Outputs a clean CSV in the exact required format: `audio_id, language, audio, option_1–5, golden_ref, wer_option1–5`.

The system must run without human intervention and must be robust to differences in audio quality, transcription formatting, and Arabic dialect variation.

---

## 3. Why Our Approach is Superior

### 3.1 No Consensus Voting, Every Signal is Anchored to the Audio

The most naive approach to this problem is majority voting: if three of five candidates share the same text, pick that one. This is mathematically unreliable. If three annotators share the same systematic transcription error, majority voting actively selects the wrong answer. Every signal in our pipeline is anchored independently to the raw audio waveform or to absolute linguistic quality. No candidate votes on another candidate.

### 3.2 Forced Alignment Directly Scores Audio-Text Correspondence

Rather than having the model generate a transcription and then compare, we use a forced aligner to measure how well each existing candidate *fits* the audio at the character level. This gives a direct audio-text correspondence score that is reference-free, it does not require any ground truth or comparison transcription to work. This is the most principled possible acoustic signal.

### 3.3 LaBSE Over GPT-2 Perplexity for Arabic

Our original proposal described a GPT-2 perplexity-based linguistic signal. During implementation, we identified a fundamental flaw in this approach: GPT-2 was trained predominantly on English text. Arabic perplexity scores from GPT-2 are unreliable because the model has no internal representation of Arabic morphology, dialects, or script. We replaced this with **LaBSE** (Language-agnostic BERT Sentence Embeddings), which was explicitly trained on 109 languages including Arabic. LaBSE provides a semantically grounded similarity score that handles Saudi dialect vocabulary naturally.

### 3.4 Whisper as Reference Generator, Not Scorer

Whisper is a powerful ASR model, but treating its output as ground truth would make our pipeline no better than "pick whatever Whisper says." Instead, we use Whisper's output purely as a *reference transcript* to anchor two comparative signals (semantic similarity and CER). When Whisper is confident, it provides a useful reference. When it is not confident, we drop it entirely and rely solely on the acoustic alignment signal. This prevents Whisper hallucinations from corrupting the selection.

### 3.5 Time Complexity

The pipeline is designed for practical execution within a hackathon window. Per-row inference time is dominated by:

- **ForcedAligner**: O(L × T) where L = candidate length and T = audio length. Text is capped at 1,000 characters to prevent O(L² × T) attention explosion from truncated candidates.
- **Whisper**: O(T) per audio clip, called once per clip (not per candidate). Result cached immediately.
- **LaBSE**: O(5 × d) where d = embedding dimensionality. Batched across all 5 candidates simultaneously.
- **CER**: O(nm) Levenshtein, negligible.

All model outputs are JSON-cached after every row. A crash at row 73 loses no work, resuming starts from row 74. The complete 100-row run on two RTX 3050 machines (50 rows each, parallel) completed within the hackathon window.

### 3.6 Parallelism Across Machines

The pipeline was deliberately designed to split by audio\_id range (`AUDIO_ID_START`, `AUDIO_ID_END` in `config.py`). Machine A ran rows 1–50, Machine B ran rows 51–100. `cross_merge.py` validated both slices and produced the final `results.csv` and `results_enhanced.csv`. This halved wall-clock time and provided redundancy against machine failure.

### 3.7 Robust to Data Corruption

The dataset contained multiple categories of corrupted or malformed transcription options (detailed in Section 5). Our pipeline identifies and flags each corruption type independently, applies appropriate penalties, and includes a hard guard (`ALL_OPTIONS_CORRUPT`) for rows where all five options are corrupted simultaneously.

---

## 4. System Architecture

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
┌─────────────────────────────────────────────────────┐
│  For each row:                                      │
│                                                     │
│  [Signal 1: Acoustic]                               │
│  Qwen3-ForcedAligner-0.6B                           │
│  → coverage ratio per candidate                     │
│  → apply format/verbosity/truncation penalties      │
│  → normalize_per_sample()                           │
│                                                     │
│  [Whisper reference]                                │
│  Whisper large-v3-turbo (called once per clip)      │
│  → ref_text, avg_logprob, no_speech_prob            │
│  → quality flag: OK / LOW_LOGPROB / NO_SPEECH /     │
│                  LOW_CONFIDENCE / ERROR             │
│                                                     │
│  [Signal 2: Semantic]                               │
│  LaBSE cosine similarity (candidate vs Whisper ref) │
│  → normalize_per_sample()                           │
│                                                     │
│  [Signal 3: CER]                                    │
│  jiwer.cer (candidate vs Whisper ref)               │
│  → normalize_per_sample()                           │
│                                                     │
│  [Fusion]                                           │
│  score = 0.40 × A + 0.30 × S + 0.30 × C             │
│  (if Whisper unreliable: score = A only)            │
│                                                     │
│  [Selection]                                        │
│  confidence_check(final, C, A_raw) →                │
│    winner_idx, ε, flag                              │
│    (if ε = 0.0: CER tiebreak → A_raw tiebreak)      │
│  if 0 < ε < 0.05: flag = ACOUSTIC_TIEBREAKER        │
│                   (winner unchanged)                │
│                                                     │
│  [WER computation]                                  │
│  jiwer.wer(golden, each candidate)                  │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  output/results_{A}_{B}.csv    │
│  (per-machine slice)           │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  cross_merge.py                │
│  • Validate both slices        │
│  • Combine, sort by audio_id   │
│  • Write results.csv (submit)  │
│  • Write results_enhanced.csv  │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  verify.py                     │
│  • 100 rows                    │
│  • Exact schema                │
│  • WER bounds [0,1]            │
│  • audio_ids 1–100 complete    │
│  • No debug columns            │
└────────────────────────────────┘
```

---

## 5. Data Preprocessing and Quality Analysis

Before any scoring, all candidate transcriptions were cleaned and flagged by `preprocess.py`. A thorough inspection of the dataset revealed the following systematic data quality issues:

### 5.1 Discovered Corruption Types

| Type | Regex | Count (approx) | Handling |
|---|---|---|---|
| Arabic diacritics (tashkeel) | `\u064B–\u065F` | 1,495+ in option\_1 alone | Stripped via `unicodedata.category` |
| Parenthetical stage directions | `\([^)]*\)` | 181 occurrences | Stripped |
| Curly-brace stage directions | `\{[^}]*\}` | 129 occurrences | Stripped |
| Embedded newlines | `\n`, `\r` | 87+ rows | Replaced with space |
| Bracket speaker labels | `\[[^\]]*\]` | 21 occurrences | Stripped |
| HEADER\_LEAK (filename in cell) | `sted Transcription` | 4 occurrences | Stripped + flagged |
| TRUNCATED (Excel 32,767-char limit) | `len(val) >= 32767` | 51 options | Penalized (× 0.7), flagged |

### 5.2 Normalization Pipeline

The `normalize_text()` function applies all cleaning steps in order. Critically, it is deterministic: calling `normalize_text(normalize_text(s)) == normalize_text(s)` for any string `s`. This is verified at runtime via a self-WER assertion (`compute_wer_cer(golden_norm, golden_norm) == 0.0`) on every processed row. If this assertion ever fires, the pipeline halts immediately, a silent non-determinism in normalization would produce wrong WER values for every row silently.

### 5.3 TRUNCATED Options

Excel imposes a 32,767-character cell limit. 51 transcription options in the dataset hit this exact limit, meaning they were silently truncated mid-sentence by the spreadsheet software. These options cannot score well against the audio because they are incomplete by definition. We apply a 0.7 acoustic penalty multiplicatively to the raw alignment score before normalization, ensuring TRUNCATED options are structurally disadvantaged but not eliminated, if all five options are truncated (which was checked), penalties cancel out in normalization and we fall back to pure raw acoustic score.

### 5.4 HEADER\_LEAK Options

Four options contained the substring `"sted Transcription"`, indicating the transcription filename leaked into the cell content. `normalize_text` strips this with a case-insensitive regex before any scoring, so the aligner never sees the leaked header. These options are flagged but receive no additional penalty beyond the semantic signal naturally reflecting their corrupted content.

### 5.5 Audio Downloading

Audio files were downloaded via HTTPS from CloudFront with 0.5s inter-request delays to avoid rate-limiting. Files were saved as `data/audio/{audio_id}.wav`. The pipeline checks for existing files before downloading, making re-runs instant on subsequent execution. All 100 audio files were downloaded successfully.

---

## 6. Scoring Methodology

### 6.1 Signal 1, Forced Acoustic Alignment (Weight: 0.40)

**Model:** Qwen3-ForcedAligner-0.6B (primary); wav2vec2-large-xlsr-53-arabic via ctc-forced-aligner (fallback if Qwen fails to load)

**Principle:** Given the audio waveform and a candidate text, forced alignment finds the time positions where each character or token in the text corresponds to a region of the audio. If a character cannot be aligned (because it was not spoken), it contributes nothing to the score.

**Score formula:**

```
A_raw = min(1.0, aligned_chars / total_chars_in_candidate)
```

Where `aligned_chars` is the sum of character lengths of all segments successfully aligned, and `total_chars_in_candidate` is the character count of the input text (spaces excluded).

**Text cap:** Candidates are capped at 1,000 characters before alignment. TRUNCATED options reach 32,767 characters, passing this to the aligner would produce a 35GB+ attention matrix and cause OOM. 1,000 characters covers approximately 100 Arabic words, well above any plausible audio clip length in this dataset.

**Penalties applied before normalization:**

- TRUNCATED flag: `A_penalized = A_raw × 0.7`
- Verbosity penalty: Exponential decay if candidate is >1.5× median length: `exp(-(R - 1.5))` where `R = candidate_len / median_len`
- Format penalty: 5% reduction for curly-brace stage directions still present in raw text; 12% reduction for speaker-label formatting (e.g., `"لينا:"` patterns)

Penalties are applied to raw scores before normalization. This preserves proportional impact, applying penalties after normalization would operate on relative rank scores [0,1] rather than acoustic confidence values.

### 6.2 Whisper Reference Generation

**Model:** openai-whisper large-v3-turbo (~1.5GB VRAM)

Whisper is called once per audio clip with `language="ar"` forced (without this, Saudi Arabic may be misdetected as Farsi). The output provides:

- `ref_text`: the model's free-decode transcription
- `avg_logprob`: average segment-level log-probability (confidence proxy)
- `no_speech_prob`: estimated probability that no speech was present

**Quality gate (AND logic):**

| Condition | Label |
|---|---|
| `avg_logprob < -1.5` AND `no_speech_prob > 0.65` | `LOW_CONFIDENCE` → drop Whisper |
| `avg_logprob < -1.5` only | `LOW_LOGPROB` → keep Whisper |
| `no_speech_prob > 0.65` only | `NO_SPEECH` → keep Whisper |
| Neither | `OK` → keep Whisper |

AND logic is used intentionally. OR logic would discard Whisper on Arabic clips that have high `no_speech_prob` due to leading silence before speech begins, a common occurrence. In our dataset, all 100 clips received quality `OK`, meaning the Whisper reference was used in all fused scores.

### 6.3 Signal 2, LaBSE Semantic Similarity (Weight: 0.30)

**Model:** sentence-transformers/LaBSE

LaBSE encodes both the Whisper reference and all five normalized candidates into a shared 768-dimensional multilingual embedding space. Cosine similarity between the reference embedding and each candidate embedding captures semantic alignment: two sentences that say the same thing in different words score high; a sentence that adds, omits, or substitutes content scores lower.

```
S_i = max(0.0, (ref_emb · cand_i_emb + 1.0) / 2.0)
```

The +1/2 shift maps cosine similarity from [-1, 1] to [0, 1]. In practice, Arabic LaBSE similarities fall in ~[0.5, 0.97], so the effective range after shifting is approximately [0.75, 0.99].

LaBSE is preferred over GPT-2 perplexity for this task because:
- It was trained on 109 languages with balanced multilingual data.
- It handles Saudi dialect Arabic natively.
- It produces a similarity score relative to the reference rather than an absolute fluency score, making it naturally calibrated for candidate ranking.

### 6.4 Signal 3, Character Error Rate (Weight: 0.30)

**Library:** jiwer

CER measures character-level Levenshtein distance between the normalized candidate and the normalized Whisper reference. We use CER rather than WER for the scoring signal because Arabic morphology attaches prefixes and suffixes to roots, a single "wrong word" in WER terms may differ from the correct form by only one or two characters.

```
C_i = max(0.0, 1.0 - min(jiwer.cer(ref_norm, cand_i_norm), 1.0))
```

Higher C means the candidate is closer to the Whisper reference at the character level.

**Note:** The CER computed here (against Whisper reference, for scoring) is distinct from the WER values in the output CSV (against the selected golden reference, for evaluation). The output WER values are computed post-selection as a quality measure of the rejected candidates.

---

## 7. Fusion, Confidence, and Selection Logic

### 7.1 Fusion Formula

For each candidate `i`:

```
score_i = W_ACOUSTIC × A_i + W_SEMANTIC × S_i + W_CER × C_i
```

Where `W_ACOUSTIC = 0.40`, `W_SEMANTIC = 0.30`, `W_CER = 0.30`.

When Whisper quality is `LOW_CONFIDENCE` (both thresholds triggered simultaneously):

```
score_i = A_i
```

This prevents a hallucinated or absent Whisper transcript from outvoting the ForcedAligner 2-to-1 via the semantic and CER signals.

### 7.2 Per-Sample Normalization

`normalize_per_sample()` in the final implementation returns the raw scores unchanged (no scaling). This is a deliberate decision reached during development: max-only normalization was found to create 1.0 ceiling ties across multiple candidates, collapsing top scores together and eliminating the epsilon signal. Raw scores preserve the natural spread of acoustic confidence values.

### 7.3 Confidence Check and Tiebreaker

```python
winner_idx = argmax(scores)
ε = scores[rank_1] - scores[rank_2]
HIGH_CONFIDENCE = ε ≥ 0.05
```

When `ε < 0.05`, two sub-cases apply:

**(a) 0 < ε < 0.05 (ACOUSTIC\_TIEBREAKER flag):** The fused argmax winner is retained as-is. No additional signal is consulted. The flag is set to `ACOUSTIC_TIEBREAKER` to record the low-margin status for downstream review.

**(b) ε = 0.0 exactly (perfect tie in fused scores):** The raw CER scores break the tie first. If those are also tied, raw acoustic scores (`A_raw`, pre-penalty and pre-normalization) decide. This logic runs entirely inside `confidence_check()` and is deterministic, no row can be left unresolved.

The tiebreaker is deterministic, runs locally, requires no API calls, and completes in microseconds.

### 7.4 WER Output Computation

Once the golden is selected, WER for all five candidates is computed via `jiwer.wer(golden_norm, candidate_norm)`. The golden candidate's own WER is hardcoded to 0.0 and verified by a runtime assertion. WER argument order: reference first, hypothesis second, denominator is reference (golden) word count.

---

## 8. Implementation Details and Engineering Decisions

### 8.1 Caching

Both Whisper transcriptions and aligner scores are cached to JSON files (`data/whisper_cache.json`, `data/aligner_cache.json`) after every single row. Cache keys for alignment scores include an MD5 hash of the first 80 characters of the candidate text. This means:

- A pipeline crash at row 73 loses zero work.
- Re-running the pipeline on already-processed rows returns cached results in milliseconds.
- The two machines (A: rows 1–50, B: rows 51–100) maintain independent caches.

### 8.2 Dual-Machine Parallelism

The pipeline was split across two machines with CUDA GPUs (RTX 3050) to reduce wall-clock time. `config.py` contains `AUDIO_ID_START` and `AUDIO_ID_END` to control which rows each machine processes. `cross_merge.py` validates both slice files for row count and schema before merging.

### 8.3 Aligner Fallback Chain

If Qwen3-ForcedAligner fails to load (import error, VRAM shortage, or corrupt weights), the pipeline automatically falls back to `CTCFallbackAligner`, which wraps `wav2vec2-large-xlsr-53-arabic` via the `ctc-forced-aligner` library. Both aligners expose the same `.align(audio, text, language)` interface, so `get_alignment_score()` is unaware of which backend is running. If both fail, a hard `RuntimeError` is raised immediately with actionable error messages, the pipeline does not proceed silently with a dead acoustic signal.

### 8.4 Empty Aligner Results

When the aligner returns an empty result (`[]` or `[[]]`), the score is set to 0.5 neutral rather than 0.0. A 0.0 score would incorrectly eliminate a valid candidate due to an aligner failure rather than a text quality failure. The 0.5 value means the acoustic signal contributes no information for that candidate on that row, but does not actively penalize it.

### 8.5 Newline Sanitization in Output

441 option cells contained embedded newlines that break CSV row boundaries when written unescaped. `_build_result()` explicitly replaces `\n` and `\r` with spaces in all raw option strings and the golden reference before writing to CSV.

### 8.6 UTF-8 BOM Encoding

All CSV files are written with `encoding='utf-8-sig'` (UTF-8 with BOM). This ensures Arabic text renders correctly when the file is opened in Excel on Windows. Without BOM, Arabic characters appear as mojibake.

### 8.7 Self-WER Assertion

Every row includes a runtime assertion:

```python
self_wer, _ = compute_wer_cer(golden_norm, golden_norm)
assert self_wer == 0.0
```

This assertion does not check the hardcoded 0.0 constant, it actually runs `normalize_text` twice on the same string and verifies the full WER pipeline end-to-end. If `normalize_text` is ever non-deterministic (producing different output on consecutive calls with the same input), this assertion catches it with a clear error message and halts the pipeline rather than writing incorrect WER values silently.

---

## 9. Results and Analysis

### 9.1 Pipeline Completion

All 100 rows were processed without errors. No rows were dropped. No rows triggered the `ALL_OPTIONS_CORRUPT` guard. Whisper produced `OK` quality transcriptions for all 100 audio clips (avg\_logprob never fell below -1.5 AND no\_speech\_prob never exceeded 0.65 simultaneously), meaning the full three-signal fusion was used for every row.

### 9.2 Accuracy Against Judge's Answer Key (Rows 1–49)

The hackathon judge released the official correct option for the first **49** audio clips (audio\_ids 1–49; audio\_id 50 has no released label). This allows a direct, ground-truth evaluation of pipeline accuracy.

**Overall Result: 39 / 49 correct, 79.6% accuracy.**

#### Methodology

For each of the 49 released rows, the judge's `correct_option` field designates the intended golden transcription. Our `predicted_option` (1–5) was compared directly to this value.

#### Summary Table

| Metric | Value |
|---|---|
| Rows evaluated | 49 (audio\_ids 1–49) |
| Correct predictions | 39 |
| Incorrect predictions | 10 |
| **Accuracy** | **79.6%** |
| HIGH\_CONFIDENCE rows (correct / total) | 4 / 5 (80.0%) |
| ACOUSTIC\_TIEBREAKER rows (correct / total) | 35 / 44 (79.5%) |

One HIGH\_CONFIDENCE row (audio\_id 1) was incorrect, the pipeline selected option 2 with a large margin (ε = 0.054), but the judge designated option 1. Nine of the ten errors occurred in the ACOUSTIC\_TIEBREAKER regime.

#### Judge Distribution vs. Our Distribution (Rows 1–49)

| Option | Judge selected | We selected |
|---|---|---|
| Option 1 | 13 | 11 |
| Option 2 | 13 | 10 |
| Option 3 | 9 | 14 |
| Option 4 | 10 | 11 |
| Option 5 | 4 | 3 |

The pipeline over-selects option 3 (14 vs 9) and under-selects options 1 and 2 relative to the judge. This suggests a systematic bias in the acoustic aligner toward option 3's formatting style on this dataset.

#### Error Analysis: The 10 Disagreements

| audio\_id | Our prediction | Judge's label | WER(judge's option vs our golden) | Confidence flag |
|---|---|---|---|---|
| 1 | Option 2 | Option 1 | 0.2667 | HIGH\_CONFIDENCE |
| 2 | Option 2 | Option 5 | 0.1071 | ACOUSTIC\_TIEBREAKER |
| 4 | Option 3 | Option 1 | 0.0345 | ACOUSTIC\_TIEBREAKER |
| 9 | Option 1 | Option 2 | 0.0000 | ACOUSTIC\_TIEBREAKER |
| 10 | Option 3 | Option 1 | 0.1429 | ACOUSTIC\_TIEBREAKER |
| 12 | Option 3 | Option 2 | 0.0476 | ACOUSTIC\_TIEBREAKER |
| 13 | Option 3 | Option 1 | 0.1304 | ACOUSTIC\_TIEBREAKER |
| 14 | Option 1 | Option 2 | 0.0889 | ACOUSTIC\_TIEBREAKER |
| 33 | Option 3 | Option 2 | 0.0938 | ACOUSTIC\_TIEBREAKER |
| 48 | Option 4 | Option 2 | 0.0233 | ACOUSTIC\_TIEBREAKER |

The "WER(judge's option vs our golden)" column measures how much the judge's designated option differs from our selected golden transcription. A value of 0.000 means both options are textually equivalent, that is a genuine ambiguity case. All other values represent genuine mismatches where the pipeline selected a textually distinct option from the judge's intended answer.

Only **1 of the 10 errors** (audio\_id 9) involves textually equivalent options (WER = 0.0 between our pick and the judge's). The remaining **9 errors are genuine mismatches**, the pipeline selected a transcription that materially differs from the judge's correct answer. This cannot be explained as label ambiguity; the pipeline made incorrect selections on these rows.

Audio\_id 1 is a notable case: despite being in HIGH\_CONFIDENCE (ε = 0.054), the pipeline's selection was wrong, indicating the forced aligner had high confidence in an incorrect alignment rather than a genuine quality difference.

#### Score Distributions: Correct vs. Incorrect Predictions

| | Correct predictions (n=39) | Incorrect predictions (n=10) |
|---|---|---|
| Mean sardines score | 0.9405 | 0.9591 |

The incorrect predictions score *higher* on average than correct ones (0.959 vs 0.941). This is a warning sign: the pipeline's internal confidence measure does not reliably correlate with external correctness on this dataset. High sardines scores do not guarantee correct option selection.

### 9.3 Confidence Distribution

| Confidence Flag | Count | Description |
|---|---|---|
| HIGH\_CONFIDENCE | 5 | ε ≥ 0.05: clear winner, large gap to runner-up |
| ACOUSTIC\_TIEBREAKER | 95 | ε < 0.05: close margin, acoustic raw scores confirm |

The high proportion of ACOUSTIC\_TIEBREAKER rows (95%) reflects the nature of this dataset. The five transcription candidates are human transcriptions of the same audio, highly similar text with small systematic differences (diacritics, formatting, minor word substitutions). The fused scores naturally cluster very close together. This is expected behavior, not a pipeline failure.

The five HIGH\_CONFIDENCE rows all had `ε ≥ 0.05`, indicating genuinely divergent transcription quality:

| audio\_id | ε | Score | Description |
|---|---|---|---|
| 1 | 0.0542 | 0.956 | Strong acoustic preference |
| 6 | 0.0792 | 0.832 | Largest margin in dataset |
| 32 | 0.0665 | 0.948 | Clear divergence |
| 34 | 0.0768 | 0.919 | Strong signal |
| 36 | 0.0702 | 0.889 | Clear winner |

### 9.4 Score Distribution

| Statistic | Fused Score (winner) |
|---|---|
| Mean | 0.9470 |
| Median | 0.9550 |
| Std dev | 0.0330 |
| Minimum | 0.832 (audio\_id 6) |
| Maximum | 0.997 |

Winners score consistently high (mean 0.947), reflecting that the selected transcriptions align strongly with all three signals simultaneously.

### 9.5 Epsilon (Confidence Margin) Distribution

| Statistic | ε |
|---|---|
| Mean | 0.0154 |
| Median | 0.0074 |
| Std dev | 0.0186 |
| Minimum | 0.0000 (29 rows) |
| Maximum | 0.0792 |

29 rows had ε = 0.0 (exact tie in fused scores). These were resolved by the CER-then-acoustic tiebreaker chain, making the final selection deterministic even in the worst case.

### 9.6 WER Analysis of Rejected Candidates

Once the golden is selected, WER is computed for all five candidates against it:

| Statistic | WER (non-golden options) |
|---|---|
| Mean WER across all rejected candidates | 0.2475 |
| Median WER | 0.0886 |
| Min WER (best rejected) | 0.0000 (near-identical twin) |
| Max WER (worst rejected) | 1.0000 |

The mean WER of 0.2475 across rejected candidates indicates that the golden transcription meaningfully differs from the rejected ones, on average, one in four words in the rejected transcriptions is in error relative to the golden. This validates that the selection is making meaningful distinctions, not arbitrarily picking among near-identical options.

Row-level analysis of WER=0 across all five options:

| Options with WER=0 per row | Number of rows |
|---|---|
| 1 option (clear winner, all others differ) | 48 |
| 2 options (near-identical transcripts) | 14 |
| 3 options (3-way near-identical cluster) | 23 |
| 4 options (only 1 outlier) | 14 |
| 5 options (all identical) | 1 |

48 rows have a single clearly distinct winner at WER=0 (all other options have WER > 0). The remaining 52 rows have some degree of near-identity across options, which is why ε values are small and the acoustic tiebreaker is needed.

### 9.7 Self-Consistency Verification

The pipeline ran `verify.py` successfully on the final `output/results.csv`:

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

---

## 10. Novel Approach and Competitive Differentiation

### 10.1 The Core Novelty: Audio-Anchored Independent Scoring

The defining architectural decision in SARDINES is that every scoring signal is anchored to something external to the candidate set. No candidate is ever used as a reference to evaluate another candidate. This is not a minor implementation detail, it is the central design principle, and it distinguishes our approach from every naive or classical transcription selection strategy.

Most teams approaching this problem will consider one or more of the following standard methods:

- **Pairwise WER cross-referencing:** Compute WER between every pair of candidates and pick the one with the lowest average WER against all others.
- **Majority voting / ROVER (Recognizer Output Voting Error Reduction):** Align all candidates at the word level and select the word chosen by the plurality of candidates at each position.
- **N-gram language model perplexity:** Score each candidate against a statistical or neural language model; pick the lowest-perplexity candidate.
- **Raw Whisper output selection:** Transcribe the audio once with Whisper and pick the candidate closest to the Whisper output by WER.

Each of these has a concrete, measurable failure mode on this specific dataset. SARDINES is designed to avoid all of them simultaneously.

### 10.2 Why Cross-Referencing Fails, and Why We Are Faster

**The correctness argument:**

Consider a dataset row where three of five annotators share the same systematic transcription error, for example, all three write "أكل الطعام" where the speaker actually said "أكل الكلام". Under pairwise cross-referencing, the erroneous version is the reference for itself and for the two other annotators who wrote the same thing. Its average WER against all five candidates will be low precisely because it has three near-identical copies agreeing with it. The two correct candidates will each have high average WER, they disagree with the majority. Cross-referencing selects the wrong answer by design.

This is not hypothetical for Gulf Arabic. The qāf/gāf alternation (ق vs. ق pronounced as /g/), the variation between Modern Standard Arabic and colloquial Gulf forms, and systematic elision patterns in fast speech all create conditions where multiple annotators from similar backgrounds will make the same error on the same word. On this 100-row dataset, these conditions are present.

**The time complexity argument:**

Cross-referencing N candidates with pairwise WER requires computing WER between every pair. For N=5 candidates and M=100 audio rows, that is:

```
Pairwise cross-referencing: O(M × N²) WER computations
= 100 × 25 = 2,500 WER computations
```

SARDINES computes one Whisper reference per audio row and scores each of the 5 candidates against that single reference:

```
SARDINES (WER/CER path): O(M × N) computations
= 100 × 5 = 500 CER/WER computations
```

This is a 5× reduction in comparison operations for the string-matching path. More importantly, the acoustic alignment path scales the same way:

```
Forced alignment: O(M × N) alignment calls
= 100 × 5 = 500 alignment operations
```

Cross-referencing with forced alignment would require aligning every candidate against every other candidate's audio-text pair, but since each candidate is text-only (not tied to a distinct audio), this path is not even well-defined for cross-referencing. The forced aligner scores text against audio, not text against text. Cross-referencing with a forced aligner would require a proxy, which reintroduces the Whisper dependency at N² scale. SARDINES uses Whisper exactly once per row, regardless of N.

Additionally, our caching architecture ensures that each (audio\_id, candidate) alignment pair is computed at most once across the entire run. If the pipeline is restarted, no work is duplicated. Cross-referencing with N² pairs would be more expensive both at first run and on any restart.

---

## 11. Robustness Across Languages and Audio Conditions

The evaluation criterion requires consistency across different languages and audio conditions. This section documents how SARDINES satisfies that criterion by design, not by accident.

### 11.1 Language Robustness

Every component in the pipeline was selected for multilingual capability rather than English-only performance:

- **Qwen3-ForcedAligner-0.6B** is a multilingual forced aligner trained on diverse language families. Its character-level alignment operates on Unicode directly, with no language-specific tokenization assumptions.
- **wav2vec2-large-xlsr-53-arabic** (CTC fallback) was explicitly fine-tuned on Arabic, providing a dedicated fallback if the primary aligner fails on Arabic-specific phonemes.
- **LaBSE** was trained on 109 languages with balanced multilingual data, meaning semantic similarity scores are calibrated equivalently across scripts and dialects, the same cosine threshold means the same thing in Arabic, Urdu, or French.
- **Whisper large-v3-turbo** supports 99 languages. The `language="ar"` parameter is set explicitly in `get_whisper_ref()` to prevent Saudi Arabic from being misdetected as Farsi, a real failure mode documented during development.
- **Text normalization** in `preprocess.py` strips Unicode diacritics via `unicodedata.category` range checks rather than hardcoded character lists, making it extensible to other diacritic-heavy scripts (Devanagari, Hebrew, Persian).

To deploy on a new language, only two lines need changing: the `language` parameter in the Whisper call and the Unicode diacritic range in `normalize_text()`. All scoring, fusion, and selection logic is language-agnostic.

### 11.2 Audio Condition Robustness

The pipeline handles degraded audio conditions at three independent layers:

**Layer 1, Whisper quality gate.** Every audio clip is assessed on `avg_logprob` and `no_speech_prob` before the Whisper transcript is used as a reference. If both thresholds are breached simultaneously (`avg_logprob < -1.5` AND `no_speech_prob > 0.65`), the pipeline drops the Whisper reference entirely and falls back to pure acoustic scoring (`score_i = A_i`). This prevents a hallucinated reference from corrupting both the semantic and CER signals on noisy audio. AND logic is used deliberately, OR logic would incorrectly drop Whisper on clips with leading silence, a common occurrence in this dataset.

**Layer 2, Forced aligner neutral fallback.** When the aligner returns an empty alignment result (which occurs on very short audio, clipped recordings, or codec artifacts), the score is set to `0.5` neutral rather than `0.0`. A `0.0` would actively penalize a valid candidate due to an audio condition failure, not a transcription quality failure. The `0.5` value removes the acoustic signal's contribution for that candidate without distorting the fused ranking.

**Layer 3, Aligner model fallback chain.** If the primary aligner (Qwen3-ForcedAligner) fails due to VRAM shortage on degraded or unusually long audio, the pipeline automatically loads `CTCFallbackAligner` (wav2vec2-large-xlsr-53-arabic) without any user intervention. Both aligners expose an identical `.align(audio, text, language)` interface. If both fail, the pipeline raises a hard `RuntimeError` with actionable instructions rather than silently proceeding with a dead acoustic signal.

Together, these three layers ensure that a single point of failure in audio quality, noisy recording, missing file, codec mismatch, leading silence, degrades the pipeline's accuracy gracefully rather than causing a crash or a silent wrong selection.

---

## 12. Known Limitations

### 12.1 Coverage Ratio Bias in Acoustic Scoring

The ForcedAligner scores candidates via `aligned_chars / total_chars`. A short candidate (e.g., 10 words) that aligns perfectly scores 1.0. A longer, complete candidate that has minor misalignments may score 0.85. This creates a structural advantage for shorter candidates when their coverage is high. We document this as a known limitation rather than attempting to correct it with heuristics that might introduce new biases. The verbosity penalty (exponential decay for candidates >1.5× median length) partially mitigates this by penalizing disproportionately long candidates, but does not fully solve the coverage ratio asymmetry.

### 12.2 LaBSE Cosine Similarity Compression

LaBSE cosine similarities for Arabic typically fall in [0.5, 0.97] rather than the full [-1, 1] range. After the +1/2 shift to [0, 1], the effective range is approximately [0.75, 0.99]. This compresses the semantic signal relative to the acoustic signal, which can range more freely across [0, 1]. In practice, this means the semantic signal contributes less discriminative power than its 0.30 weight implies. A calibration step mapping LaBSE scores to a wider range would improve signal quality.

### 12.3 Whisper as Reference Generator

Our pipeline uses Whisper to generate a reference transcript that anchors the semantic and CER signals. If Whisper systematically misrecognizes a particular speaker style, accent feature, or vocabulary domain across multiple clips, both the semantic and CER signals will be biased toward Whisper's preferred transcription style rather than the ground-truth spoken content. In this dataset, Whisper was `OK` quality on all 100 rows, but this limitation would surface on noisier data.

### 12.4 Single-Language Dataset

The pipeline was designed and tested on Arabic (Saudi dialect) exclusively. While LaBSE and Qwen3-ForcedAligner both support other languages, the text normalization pipeline in `preprocess.py` is Arabic-specific (diacritic stripping Unicode range, Arabic script preservation in regex). A multi-language deployment would require language-conditional normalization branches.

### 12.5 Confidence Threshold Calibration

The CONFIDENCE\_THRESHOLD = 0.05 was set empirically. In this dataset, 95% of rows fall below this threshold, meaning almost all selections go through the acoustic tiebreaker path. The threshold is appropriate for a dataset of near-identical human transcriptions, but may need recalibration for datasets with more divergent candidate quality.

---

## 13. Future Improvements

### 13.1 Calibrated Confidence Scoring

The current epsilon metric measures raw score gap, which is sensitive to absolute score magnitude. A calibrated confidence measure (e.g., temperature-scaled softmax over fused scores, or a Platt-scaled probability) would make the confidence flag more interpretable and allow principled threshold selection.

### 13.2 Language-Adaptive Text Normalization

The current normalization strips Arabic diacritics, parenthetical directions, speaker labels, and stage directions, all of which are specific to this dataset's transcription style. A production system should auto-detect formatting conventions from the data rather than hardcode them, using lightweight pattern detection to identify and remove dataset-specific artifacts.

### 13.3 Dynamic Fusion Weight Optimization

The weights (0.40 acoustic, 0.30 semantic, 0.30 CER) were set based on domain knowledge and pilot testing. With a labeled validation subset (even 10–15 annotated rows with known golden transcriptions), these weights could be optimized via grid search or gradient-free optimization to maximize selection accuracy. The `fuse()` function accepts weights as parameters, so this would require no structural change to the codebase.

### 13.4 Ensemble Alignment Models

Running two or more forced aligners (e.g., Qwen3 and wav2vec2-xlsr) and averaging their alignment scores would reduce the impact of any single model's failure modes. Currently the fallback chain runs them sequentially, an ensemble would run them in parallel and combine scores.

### 13.5 Direct WER/CER Against All Pairs

Rather than computing CER only against the Whisper reference, one could compute pairwise CER between all 5 candidates to identify the consensus transcription, the one with minimum average edit distance to all others. This partially reintroduces consensus logic, but does so at the character level (CER-based) rather than binary match voting, making it more robust to shared errors. Combined with the acoustic signal as a veto, this could improve accuracy on rows where Whisper is unreliable.

### 13.6 Segment-Level Acoustic Scoring

The current pipeline scores each candidate against the entire audio clip. For long audio clips with multiple speaker turns, segment-level alignment (scoring each sentence against the corresponding audio segment) would produce more fine-grained acoustic evidence. This requires a diarization step to locate speaker boundaries, which was out of scope for the hackathon window but is straightforward to add.

### 13.7 Parallel Per-Row Processing

Currently, rows are processed sequentially. With a GPU cluster or batched inference, the ForcedAligner and LaBSE steps could be parallelized across rows. The JSON cache is row-addressable, so parallel writes require only row-level locking rather than file-level locking.

---

## 14. Reproducibility and Submission Checklist

### Technology Stack

| Component | Library/Model | Purpose |
|---|---|---|
| ForcedAligner (primary) | Qwen3-ForcedAligner-0.6B (`qwen-asr`) | Acoustic coverage scoring |
| ForcedAligner (fallback) | wav2vec2-large-xlsr-53-arabic (`ctc-forced-aligner`) | Acoustic fallback |
| Whisper | openai-whisper large-v3-turbo | Reference transcription + quality signal |
| Semantic similarity | sentence-transformers/LaBSE | Multilingual embedding similarity |
| WER/CER | jiwer | Character and word error rate |
| Audio loading | torchaudio | 16kHz mono WAV preprocessing |
| Data handling | pandas | CSV I/O, utf-8-sig encoding |
| Deep learning | torch, transformers, accelerate | Model inference |

### Reproducibility Steps

> **Before running anything:** Refer to [ALL_PREREQUISITES.txt](../ALL_PREREQUISITES.txt) and ensure every software, system-level, and hardware requirement listed there is installed on your machine.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place input CSV at data/input.csv

# 3. Configure machine slice in config.py:
#    AUDIO_ID_START = 1
#    AUDIO_ID_END = 100

# 4. Download Qwen3-ForcedAligner-0.6B into ./Qwen3-ForcedAligner-0.6B/

# 5. Run pipeline
python pipeline.py

# 6. Merge slices (if two machines were used)
python cross_merge.py

# 7. Verify output
python verify.py
```

### Final Output Files

| File | Description |
|---|---|
| `output/results.csv` | **Submission file**, 100 rows, exact required schema |
| `output/results_enhanced.csv` | Debug file, includes scores, confidence flags, epsilon |
| `data/whisper_cache.json` | Whisper output cache (100 entries) |
| `data/aligner_cache.json` | Alignment score cache (up to 500 entries) |

---

*sardines, Hackenza 2026, BITS Pilani, KK Birla Goa Campus*
*Ansh Varma · Sana H · Darshan Rajagoli · Rashida Baldiwala*


