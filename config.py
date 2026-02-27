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