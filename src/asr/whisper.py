import subprocess

import numpy as np
import torch
from transformers import pipeline

DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000


def load_audio(path_or_bytes):
    source = ["-i", "pipe:0"] if isinstance(path_or_bytes, (bytes, bytearray)) else ["-i", str(path_or_bytes)]
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", *source, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        input=path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else None,
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


class Transcriber:
    def __init__(self, model=DEFAULT_MODEL, batch_size=16):
        self.model = model
        self.batch_size = batch_size
        self.pipe = pipeline("automatic-speech-recognition", model=model, dtype=torch.float16, device="cuda")

    def __call__(self, audios):
        inputs = [{"raw": a, "sampling_rate": SAMPLE_RATE} for a in audios]
        out = self.pipe(inputs, batch_size=self.batch_size, generate_kwargs={"language": "russian", "task": "transcribe"})
        return [o["text"].strip() for o in out]
