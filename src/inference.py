"""Load an exported model folder (as published on the Hugging Face Hub) and translate.

    from fr_en_transformer import load_translator      # inside the HF repo
    tr = load_translator("path/to/snapshot")
    tr.translate(["Ne t'inquiète pas !"])
"""
import json
import os

import sentencepiece as spm
import torch
from safetensors.torch import load_file

from .decode import Translator
from .model import Seq2SeqTransformer


def load_translator(path, device=None, **decode_overrides):
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    model = Seq2SeqTransformer(**cfg["model"])
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(path, "spm.model"))
    return Translator(model.to(device), sp, device, **{**cfg["decode"], **decode_overrides})
