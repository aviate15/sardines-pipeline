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

    # Strip (يضحك) (متوترة) parenthetical stage directions — 181 confirmed occurrences
    text = re.sub(r'\([^)]*\)', '', text)

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