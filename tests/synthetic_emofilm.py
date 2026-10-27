"""A small stand-in for data/emofilm/assembled (affect.data.emofilm, added 2026-10-20): real 16 kHz WAV files named as
EmoFilm's clips, with a MANIFEST.csv and ASSEMBLY.json written as assemble_emofilm.py writes them, so the dataset and
the cold test run without the corpus. Each clip's samples carry its emotion in their level."""
import csv
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

from affect.data.emofilm import COLUMNS

EMOTIONS = {"gio": "happiness", "rab": "anger", "tri": "sadness"}
LEVEL = {"gio": 3000, "rab": 6000, "tri": 1000}


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.astype("<i2").tobytes())


def write_assembly(assembled: Path, per_cell: int = 2, seed: int = 0) -> list:
    """per_cell clips for every language, gender and kept emotion; returns the manifest's rows."""
    rng = np.random.default_rng(seed)
    (assembled / "wav").mkdir(parents=True)
    rows, k = [], 0
    for lang in ("en", "es", "it"):
        for gender in ("f", "m"):
            for code, emotion in EMOTIONS.items():
                for i in range(per_cell):
                    k += 1
                    stem = f"{gender}_{code}{k:03d}a{lang}"
                    n = 1600 + 80 * (k % 7)
                    samples = rng.normal(0, LEVEL[code], size=n).clip(-32768, 32767)
                    write_wav(assembled / "wav" / f"{stem}.wav", samples)
                    digest = hashlib.sha256((assembled / "wav" / f"{stem}.wav").read_bytes()).hexdigest()
                    rows.append({"utt_id": stem, "audio_path": f"wav/{stem}.wav",
                                 "speaker": f"emofilm_{gender}_{1 + k % 5:03d}", "film": f"film_{1 + k % 4:02d}",
                                 "language": lang, "gender": gender, "emotion_code": code, "emotion": emotion,
                                 "scene": f"{k:03d}", "segment": "a", "n_frames": n, "sample_rate": 16000,
                                 "sha256": digest, "source_sha256": digest})
    with open(assembled / "MANIFEST.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest_sha = hashlib.sha256((assembled / "MANIFEST.csv").read_bytes()).hexdigest()
    (assembled / "ASSEMBLY.json").write_text(json.dumps({"dataset": "emofilm", "kept": len(rows),
                                                         "manifest_sha256": manifest_sha}) + "\n")
    return rows
