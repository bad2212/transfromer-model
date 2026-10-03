"""Export the final model to a self-contained folder and push it to the Hugging Face Hub.

    python -m src.hub --config configs/base_6x3.yaml --out_dir <ckpt dir> --repo Badalt/fr-en-transformer-scratch [--push]

The folder holds model.safetensors (averaged weights), config.json (architecture + selected decoding
settings), spm.model, the inference code as a small package (fr_en_transformer/), and README.md
(model card, metrics filled from <out_dir>/eval/dev_report.json and analysis/analysis.json).
"""
import argparse
import glob
import json
import os
import shutil

from safetensors.torch import save_file

from .predict import load_model
from .utils import load_config

CODE = ["model.py", "decode.py", "utils.py", "inference.py"]


def fmt_slices(rep):
    rows = ["| slice | n | BLEU | chrF |", "|---|---|---|---|"]
    for k in ["seen", "long", "unseen_domain"]:
        v = rep["by_slice"].get(k)
        if v:
            rows.append(f"| {k} | {v['n']} | {v['bleu']:.2f} | {v['chrf']:.2f} |")
    rows.append(f"| **all** | {rep['all']['n']} | {rep['all']['bleu']:.2f} | {rep['all']['chrf']:.2f} |")
    return "\n".join(rows)


def model_card(repo, cfg, rep, n_params, links, analysis):
    m, d = cfg["model"], rep["decode"]
    stats = links.get("data_stats", {})
    extra = ""
    if analysis:
        extra = "\n### Larger eval-only sets (post-hoc, not used for selection)\n\n| set | n | BLEU | chrF |\n|---|---|---|---|\n"
        for name, v in analysis.get("aux_sets", {}).items():
            extra += f"| {name} | {v['n']} | {v['bleu']:.2f} | {v['chrf']:.2f} |\n"
    return f"""---
language: [fr, en]
license: mit
tags: [translation, transformer, from-scratch, seq2seq]
datasets: [Helsinki-NLP/opus-100]
metrics: [bleu, chrf]
pipeline_tag: translation
---

# French → English Transformer (trained from scratch)

An encoder-decoder Transformer trained **from scratch** (no pretrained weights) for fr→en translation,
built for a generalization study: in-domain vs long sentences vs an unseen domain (literature).

Code, training and the full write-up: {links.get('github', '_TBD_')}. Training curves: {links.get('wandb', '_TBD_')}.

## Architecture

| | |
|---|---|
| Type | encoder-decoder, pre-LayerNorm (Xiong et al. 2020) |
| Layers | {m['enc_layers']} encoder / {m['dec_layers']} decoder (deep-encoder/shallow-decoder, Kasai et al. 2021) |
| Width | d_model {m['d_model']}, {m['n_heads']} heads, FFN {m['d_ff']}, ReLU, dropout {m['dropout']} |
| Positions | {'rotary (RoPE, Su et al. 2021) in self-attention' if m['pos'] == 'rope' else 'sinusoidal'} |
| Embeddings | one matrix shared by encoder input, decoder input and output layer (Press & Wolf 2017) |
| Parameters | {n_params / 1e6:.1f}M |
| Tokenizer | SentencePiece unigram, {cfg['tokenizer']['vocab_size']} joint fr+en pieces, byte fallback; source-side subword sampling α={cfg['tokenizer']['src_sampling_alpha']} during training |
| Decoding | beam {d['beam']}, GNMT length penalty α={d['alpha']}; inputs over {cfg['decode']['split_over_tokens']} tokens are split at sentence boundaries |

## Training data

`Helsinki-NLP/opus-100` (en-fr), `train` split, fr→en. After normalisation (NFKC, typographic quotes to
ASCII), filtering (empty, source = target, length ratio outside [{cfg['data']['min_char_ratio']}, {cfg['data']['max_char_ratio']}],
over {cfg['data']['max_chars']} chars), deduplication and removal of any pair whose source appears in the
dev/test sets: **{stats.get('kept', '?'):,} of {stats.get('input', '?'):,} pairs** kept. `opus_books` was never used
for training or model selection. Label smoothing 0.1, AdamW, warmup {cfg['train']['warmup_steps']} + inverse-sqrt, ~24k tokens/update,
fp16 on one T4; final weights = average of the last checkpoints.

## Results (provided dev set, official `score.py`, lowercased)

{fmt_slices(rep)}

**OVERALL** (0.4·BLEU + 0.4·chrF + 0.2·chrF on unseen_domain) = **{rep['OVERALL']:.2f}**
{extra}
Dev slices are small (60/30/60 sentences), so per-slice numbers carry wide confidence intervals (see the report).

## Usage

```python
from huggingface_hub import snapshot_download
import sys
path = snapshot_download("{repo}")
sys.path.insert(0, path)
from fr_en_transformer import load_translator   # needs: torch, sentencepiece, safetensors
tr = load_translator(path)                     # beam={d['beam']}, alpha={d['alpha']}; override e.g. beam=1
print(tr.translate(["Ne t'inquiète pas !", "Il jeta son chapeau par terre."]))
```

## Limitations

- Trained only on opus-100 (largely subtitles, web and official texts), so it does best on short, conversational
  or administrative sentences. Literary French (archaic vocabulary, long dependencies) is clearly weaker: see the
  unseen_domain slice.
- Names and rare words are split into subwords or bytes and are often mistranslated or transliterated.
- Single-reference automatic metrics with the challenge tokenisation; not comparable to sacreBLEU.
- fr→en only; outputs keep case but use ASCII quotes.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir")
    ap.add_argument("--repo", required=True, help="e.g. Badalt/fr-en-transformer-scratch")
    ap.add_argument("--export_dir", default="export")
    ap.add_argument("--github", default="https://github.com/bad2212/transfromer-model")
    ap.add_argument("--wandb", default="")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out_dir or cfg["out_dir"]
    rep = json.load(open(f"{out}/eval/dev_report.json"))
    paths = rep["decode"]["checkpoints"]
    paths = [p if os.path.exists(p) else os.path.join(out, os.path.basename(p)) for p in paths]
    model = load_model(paths, "cpu")
    n_params = sum(p.numel() for p in model.parameters())

    ex = args.export_dir
    shutil.rmtree(ex, ignore_errors=True)
    os.makedirs(f"{ex}/fr_en_transformer")
    # embed is tied to the output projection; safetensors stores each tensor once
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, f"{ex}/model.safetensors")
    decode = {"beam": rep["decode"]["beam"], "len_penalty": rep["decode"]["alpha"],
              "max_len_a": cfg["decode"]["max_len_a"], "max_len_b": cfg["decode"]["max_len_b"],
              "split_over_tokens": cfg["decode"]["split_over_tokens"]}
    json.dump({"model": model.config, "decode": decode, "training_config": cfg, "dev_report": rep},
              open(f"{ex}/config.json", "w"), indent=1)
    shutil.copy(f"{out}/spm.model", f"{ex}/spm.model")
    src = os.path.dirname(os.path.abspath(__file__))
    for f in CODE:
        shutil.copy(os.path.join(src, f), f"{ex}/fr_en_transformer/{f}")
    with open(f"{ex}/fr_en_transformer/__init__.py", "w") as f:
        f.write("from .inference import load_translator  # noqa: F401\n")
    for f in ["dev_report.json", "sweep.json", "test_predictions.json"]:
        if os.path.exists(f"{out}/eval/{f}"):
            shutil.copy(f"{out}/eval/{f}", f"{ex}/{f}")

    analysis = None
    if os.path.exists(f"{out}/analysis/analysis.json"):
        analysis = json.load(open(f"{out}/analysis/analysis.json"))
        shutil.copytree(f"{out}/analysis", f"{ex}/analysis")
    stats_path = os.path.join(cfg["work_dir"], "data_stats.json")
    links = {"github": args.github, "wandb": args.wandb,
             "data_stats": json.load(open(stats_path)) if os.path.exists(stats_path) else {}}
    with open(f"{ex}/README.md", "w") as f:
        f.write(model_card(args.repo, cfg, rep, n_params, links, analysis))
    print(f"exported to {ex}/: {sorted(os.listdir(ex))}")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=ex, repo_id=args.repo, commit_message="Upload from-scratch fr-en Transformer")
        print(f"pushed: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
