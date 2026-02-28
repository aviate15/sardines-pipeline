# ── Models ──────────────────────────────────────────────────────────
WHISPER_MODEL      = "large-v3-turbo"
ALIGNER_MODEL_PATH = "./Qwen3-ForcedAligner-0.6B"
LABSE_MODEL        = "sentence-transformers/LaBSE"

# ── Fusion weights ───────────────────────────────────────────────────
W_ACOUSTIC = 0.40
W_SEMANTIC = 0.30
W_CER      = 0.30

# ── Confidence gate ──────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.05

# ── Whisper quality gates ─────────────────────────────────────────────
LOGPROB_THRESHOLD   = -1.5
NO_SPEECH_THRESHOLD = 0.65

# ── Paths ─────────────────────────────────────────────────────────────
AUDIO_DIR     = "data/audio/"
INPUT_CSV     = "data/input.csv"
OUTPUT_CSV    = "output/results.csv"
WHISPER_CACHE = "data/whisper_cache.json"
ALIGNER_CACHE = "data/aligner_cache.json"

# ── API — NEVER COMMIT. Share via WhatsApp only. ──────────────────────
ANTHROPIC_API_KEY = "sk-ant-PASTE_HERE"

# ── THIS MACHINE'S SLICE ──────────────────────────────────────────────
# Machine A:  START=1,  END=50
# Machine B:  START=51, END=100
# Machine C/D: not running pipeline
AUDIO_ID_START = 1
AUDIO_ID_END   = 50