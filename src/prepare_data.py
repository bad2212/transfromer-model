"""Download opus-100 en-fr, normalise, filter, dedupe, and train the SentencePiece model.

    python -m src.prepare_data --config configs/base.yaml

Writes to <work_dir>/:  train.fr train.en valid.fr valid.en  spm.model  data_stats.json
opus_books is never touched here (it must not be trained on).
"""
import argparse
import json
import os
import random
import re

import sentencepiece as spm
from datasets import load_dataset

from .utils import load_config, normalize, read_jsonl, seed_everything

_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def filter_pairs(pairs, cfg, banned_sources):
    """Returns kept pairs and a counter of why the rest were dropped (for the report)."""
    dc = cfg["data"]
    stats = {"input": len(pairs)}
    seen, kept = set(), []

    def drop(reason):
        stats[reason] = stats.get(reason, 0) + 1

    for fr, en in pairs:
        if not fr or not en or not _LETTER.search(fr) or not _LETTER.search(en):
            drop("empty_or_no_letters")
        elif fr.lower() == en.lower():
            drop("src_equals_tgt")
        elif max(len(fr), len(en)) > dc["max_chars"]:
            drop("too_long_chars")
        elif not dc["min_char_ratio"] <= len(en) / len(fr) <= dc["max_char_ratio"]:
            drop("bad_length_ratio")
        elif fr in banned_sources:
            drop("overlaps_dev_or_test")
        elif (fr, en) in seen:
            drop("duplicate")
        else:
            seen.add((fr, en))
            kept.append((fr, en))
    stats["kept"] = len(kept)
    return kept, stats


def write_lines(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def train_spm(cfg, work):
    tc = cfg["tokenizer"]
    spm.set_random_generator_seed(cfg["seed"])
    spm.SentencePieceTrainer.train(
        input=[f"{work}/train.fr", f"{work}/train.en"],
        model_prefix=f"{work}/spm",
        vocab_size=tc["vocab_size"],
        model_type=tc["model_type"],
        input_sentence_size=tc["sample_sentences"],
        shuffle_input_sentence=True,
        character_coverage=0.9995,
        byte_fallback=True,              # unseen characters become bytes, never <unk>
        normalization_rule_name="identity",  # we normalise ourselves (src/utils.normalize)
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        seed_sentencepiece_size=1000000,
        num_threads=2,  # fixed: unigram EM results depend on the thread count (matches the Colab run)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)

    ds = load_dataset("Helsinki-NLP/opus-100", "en-fr")
    norm = lambda split: [(normalize(t["fr"]), normalize(t["en"])) for t in ds[split]["translation"]]

    banned = {normalize(r["source"]) for p in ["data/dev/inputs.jsonl", "data/test/inputs.jsonl"]
              for r in read_jsonl(p)}
    train, stats = filter_pairs(norm("train"), cfg, banned)

    if cfg["data"]["max_pairs"] and len(train) > cfg["data"]["max_pairs"]:
        random.Random(cfg["seed"]).shuffle(train)
        train = train[: cfg["data"]["max_pairs"]]
        stats["subsampled_to"] = len(train)

    valid = norm("validation")
    write_lines(f"{work}/train.fr", [p[0] for p in train])
    write_lines(f"{work}/train.en", [p[1] for p in train])
    write_lines(f"{work}/valid.fr", [p[0] for p in valid])
    write_lines(f"{work}/valid.en", [p[1] for p in valid])

    train_spm(cfg, work)
    with open(f"{work}/data_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
