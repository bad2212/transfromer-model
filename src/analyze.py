"""Generalization analysis for the report. Post-hoc only: nothing here feeds back into training or
model/decoding selection (PLAN.md D4).

    python -m src.analyze --config configs/base_6x3.yaml [--out_dir ...] [--n_aux 1000]

Uses the checkpoints and decoding settings recorded in <out_dir>/eval/dev_report.json and writes
<out_dir>/analysis/: analysis.json, examples.md, length.png, rare.png

1. dev scores by slice with 95% bootstrap CIs, plus the seen - unseen_domain gap CI
2. larger eval-only sets: opus-100 `test` (in-domain) and an opus_books sample (unseen domain;
   sentences that appear in our dev/test are removed)
3. chrF/BLEU by source length bucket, per domain
4. rare-word effect: sentence chrF vs share of source words that are rare in training
5. domain statistics: length, OOV/rare-word rates, and the model's per-token loss on the reference
   (forced decoding), which separates "harder to translate" from "metric/reference noise"
6. long-input splitting on vs off, for inputs above the split threshold
7. worst examples per slice
"""
import argparse
import json
import math
import os
import random
import re
from collections import Counter

import sentencepiece as spm
import torch
import torch.nn.functional as F
from datasets import load_dataset

from .decode import Translator
from .metrics import official, score_predictions
from .model import BOS, EOS, PAD
from .predict import load_model
from .utils import get_device, load_config, normalize, read_jsonl, write_json

WORD = re.compile(r"\w+", re.UNICODE)
LEN_BUCKETS = [(1, 10), (11, 20), (21, 30), (31, 45), (46, 10 ** 6)]


def bootstrap(hyps, refs, n=1000, seed=0):
    rng = random.Random(seed)
    idx = range(len(refs))
    bleus, chrfs = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        r = official.score_slice([hyps[i] for i in s], [refs[i] for i in s])
        bleus.append(r["bleu"])
        chrfs.append(r["chrf"])
    ci = lambda xs: [sorted(xs)[int(0.025 * n)], sorted(xs)[int(0.975 * n) - 1]]
    return {"bleu_ci": ci(bleus), "chrf_ci": ci(chrfs), "_bleus": bleus, "_chrfs": chrfs}


def word_freqs(path):
    c = Counter()
    with open(path) as f:
        for line in f:
            c.update(w.lower() for w in WORD.findall(line))
    return c


def rare_share(text, freqs, thresh):
    words = [w.lower() for w in WORD.findall(text)]
    return sum(freqs[w] < thresh for w in words) / max(len(words), 1)


@torch.no_grad()
def ref_loss(model, sp, device, pairs, bs=64):
    """Mean per-token cross-entropy of the reference under the model (teacher forcing)."""
    tot, n = 0.0, 0
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        S = [sp.encode(normalize(s))[:255] + [EOS] for s, _ in chunk]
        T = [[BOS] + sp.encode(normalize(t))[:255] + [EOS] for _, t in chunk]
        src = torch.full((len(chunk), max(map(len, S))), PAD, dtype=torch.long)
        tgt = torch.full((len(chunk), max(map(len, T))), PAD, dtype=torch.long)
        for j, (s, t) in enumerate(zip(S, T)):
            src[j, :len(s)], tgt[j, :len(t)] = torch.tensor(s), torch.tensor(t)
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1]).float()
        tot += F.cross_entropy(logits.transpose(1, 2), tgt[:, 1:], ignore_index=PAD, reduction="sum").item()
        n += (tgt[:, 1:] != PAD).sum().item()
    return tot / n


def bucket_scores(rows, key, buckets, label):
    out = []
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= key(r) <= hi]
        if len(sel) >= 5:
            s = official.score_slice([r["hyp"] for r in sel], [r["ref"] for r in sel])
            out.append({"bucket": label(lo, hi), "n": len(sel), "bleu": s["bleu"], "chrf": s["chrf"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir")
    ap.add_argument("--n_aux", type=int, default=1000, help="sentences per eval-only set")
    ap.add_argument("--rare_thresh", type=int, default=5, help="training count below which a word is rare")
    ap.add_argument("--aux_beam", type=int, help="beam for the eval-only sets (default: the dev setting); "
                                                 "1 = greedy, ~5x cheaper in time and memory")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = args.out_dir or cfg["out_dir"]
    dst = f"{out}/analysis"
    os.makedirs(dst, exist_ok=True)
    device = get_device()

    rep = json.load(open(f"{out}/eval/dev_report.json"))
    d = rep["decode"]
    paths = [p if os.path.exists(p) else os.path.join(out, os.path.basename(p)) for p in d["checkpoints"]]
    model = load_model(paths, device)
    sp = spm.SentencePieceProcessor(model_file=f"{out}/spm.model")
    dc = cfg["decode"]
    aux_beam = args.aux_beam or d["beam"]
    aux_alpha = d["alpha"] if aux_beam > 1 else 0.0
    tr = Translator(model, sp, device, beam=aux_beam, len_penalty=aux_alpha, max_len_a=dc["max_len_a"],
                    max_len_b=dc["max_len_b"], split_over_tokens=dc["split_over_tokens"])
    freqs = word_freqs(f"{cfg['work_dir']}/train.fr")
    result = {"decode": d, "aux_decode": {"beam": aux_beam, "alpha": aux_alpha}, "rare_thresh": args.rare_thresh}

    # 1. dev with bootstrap CIs
    dev_in = {r["id"]: r for r in read_jsonl("data/dev/inputs.jsonl")}
    dev_gold = read_jsonl("data/dev/labels.jsonl")
    dev_pred = json.load(open(f"{out}/eval/dev_predictions.json"))
    dev_rows = [{"id": g["id"], "slice": g["slice"], "src": dev_in[g["id"]]["source"], "ref": g["reference"],
                 "hyp": dev_pred[g["id"]]} for g in dev_gold]
    result["dev"] = score_predictions(dev_pred, dev_gold)
    boots = {}
    for sl in ["seen", "long", "unseen_domain"]:
        rs = [r for r in dev_rows if r["slice"] == sl]
        boots[sl] = bootstrap([r["hyp"] for r in rs], [r["ref"] for r in rs])
        result["dev"]["by_slice"][sl].update({k: v for k, v in boots[sl].items() if not k.startswith("_")})
    gap_b = sorted(a - b for a, b in zip(boots["seen"]["_bleus"], boots["unseen_domain"]["_bleus"]))
    gap_c = sorted(a - b for a, b in zip(boots["seen"]["_chrfs"], boots["unseen_domain"]["_chrfs"]))
    result["dev"]["gap_seen_minus_unseen"] = {"bleu_ci": [gap_b[25], gap_b[974]], "chrf_ci": [gap_c[25], gap_c[974]]}

    # 2. eval-only sets
    banned = {normalize(r["source"]) for p in ["data/dev/inputs.jsonl", "data/test/inputs.jsonl"] for r in read_jsonl(p)}
    rng = random.Random(cfg["seed"])
    o100 = [(t["fr"], t["en"]) for t in load_dataset("Helsinki-NLP/opus-100", "en-fr", split="test")["translation"]]
    books = [(t["fr"], t["en"]) for t in load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")["translation"]]
    aux = {}
    for name, pairs in [("opus100_test", o100), ("opus_books_sample", books)]:
        pairs = [p for p in pairs if normalize(p[0]) not in banned and p[0].strip() and p[1].strip()]
        pairs = rng.sample(pairs, min(args.n_aux, len(pairs)))
        cache = f"{dst}/{name}_beam{aux_beam}_predictions.json"   # decoding is the slow part: reuse on rerun
        if os.path.exists(cache) and [r["src"] for r in json.load(open(cache))] == [s for s, _ in pairs]:
            aux[name] = json.load(open(cache))
            print(f"reused {name}: {len(pairs)}", flush=True)
            continue
        hyps = tr.translate([s for s, _ in pairs])
        aux[name] = [{"src": s, "ref": t, "hyp": h} for (s, t), h in zip(pairs, hyps)]
        write_json(aux[name], cache)
        print(f"decoded {name}: {len(pairs)}", flush=True)
    result["aux_sets"] = {k: official.score_slice([r["hyp"] for r in v], [r["ref"] for r in v]) for k, v in aux.items()}

    # 3./4. length and rare-word buckets, per domain
    nwords = lambda r: len(r["src"].split())
    result["by_length"], result["by_rare"], result["domain_stats"] = {}, {}, {}
    rare_buckets = [(0, 0), (1e-9, 0.1), (0.1 + 1e-9, 1.0)]   # (0, 0.1] and (0.1, 1]: disjoint
    for name, rows in aux.items():
        result["by_length"][name] = bucket_scores(rows, nwords, LEN_BUCKETS,
                                                  lambda lo, hi: f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+")
        for r in rows:
            r["rare"] = rare_share(r["src"], freqs, args.rare_thresh)
        result["by_rare"][name] = bucket_scores(rows, lambda r: r["rare"], rare_buckets,
                                                lambda lo, hi: "none" if hi == 0 else ("up to 10%" if hi <= 0.1 else "over 10%"))
        words = [w.lower() for r in rows for w in WORD.findall(r["src"])]
        result["domain_stats"][name] = {
            "mean_src_words": sum(map(nwords, rows)) / len(rows),
            "oov_token_rate": sum(freqs[w] == 0 for w in words) / len(words),
            "rare_token_rate": sum(freqs[w] < args.rare_thresh for w in words) / len(words),
            "sent_with_rare_word": sum(r["rare"] > 0 for r in rows) / len(rows),
            "ref_loss_per_token": ref_loss(model, sp, device, [(r["src"], r["ref"]) for r in rows]),
            "hyp_ref_len_ratio": sum(len(r["hyp"].split()) for r in rows) / sum(len(r["ref"].split()) for r in rows),
        }
        result["domain_stats"][name]["ref_ppl"] = math.exp(result["domain_stats"][name]["ref_loss_per_token"])

    # 6. split on/off for inputs above the threshold
    long_rows = [r for rows in aux.values() for r in rows if len(sp.encode(normalize(r["src"]))) > dc["split_over_tokens"]]
    if long_rows:
        nosplit = Translator(model, sp, device, beam=aux_beam, len_penalty=aux_alpha, max_len_a=dc["max_len_a"],
                             max_len_b=dc["max_len_b"], split_over_tokens=10 ** 9)
        h2 = nosplit.translate([r["src"] for r in long_rows])
        refs = [r["ref"] for r in long_rows]
        result["split_ablation"] = {"n": len(long_rows),
                                    "split": official.score_slice([r["hyp"] for r in long_rows], refs),
                                    "no_split": official.score_slice(h2, refs)}

    # 7. worst examples
    with open(f"{dst}/examples.md", "w") as f:
        for sl in ["seen", "long", "unseen_domain"]:
            rs = sorted((r for r in dev_rows if r["slice"] == sl), key=lambda r: official.chrf_sentence(r["hyp"], r["ref"]))
            f.write(f"## {sl}: lowest / highest sentence chrF\n\n")
            for r in rs[:4] + rs[-2:]:
                f.write(f"- **{official.chrf_sentence(r['hyp'], r['ref']):.0f}** `{r['id']}`\n  - SRC: {r['src']}\n"
                        f"  - REF: {r['ref']}\n  - HYP: {r['hyp']}\n")
            f.write("\n")

    write_json(result, f"{dst}/analysis.json")
    for name, rows in aux.items():
        write_json(rows, f"{dst}/{name}_predictions.json")
    plot(result, dst)
    print(json.dumps({k: result[k] for k in ["aux_sets", "domain_stats"]}, indent=1))
    print(f"wrote {dst}/")


def plot(result, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, fname, xlabel in [("by_length", "length.png", "source length (words)"),
                               ("by_rare", "rare.png", "share of rare source words")]:
        fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=200)
        for name, color in [("opus100_test", "#1f6feb"), ("opus_books_sample", "#d1495b")]:
            rows = result[key].get(name, [])
            ax.plot([r["bucket"] for r in rows], [r["chrf"] for r in rows], marker="o", color=color,
                    label="opus-100 test (in-domain)" if name == "opus100_test" else "opus_books (unseen domain)")
            for r in rows:
                ax.annotate(f"n={r['n']}", (r["bucket"], r["chrf"]), fontsize=5, xytext=(0, 4),
                            textcoords="offset points", ha="center", color=color)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("chrF", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{dst}/{fname}")
        plt.close(fig)


if __name__ == "__main__":
    main()
