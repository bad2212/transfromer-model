"""Small shared helpers: config, seeding, device, text normalisation, I/O."""
import json
import os
import random
import re
import unicodedata

import numpy as np
import torch


def load_config(path):
    import yaml  # training-only dependency; inference doesn't need it
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Typographic punctuation -> ASCII. The books domain uses ’ « » while most of
# opus-100 and all books *references* use straight quotes; scorer treats ’ and '
# as different tokens.
_PUNCT_MAP = str.maketrans({
    "’": "'", "‘": "'", "‚": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-",
})


def normalize(text):
    """Applied to source text before tokenization and to model output after decoding."""
    text = unicodedata.normalize("NFKC", text or "")  # also maps NBSP -> space, … -> ...
    text = re.sub(r"«\s*", '"', text)             # « bonjour » -> "bonjour"
    text = re.sub(r"\s*»", '"', text)
    text = text.translate(_PUNCT_MAP)
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
