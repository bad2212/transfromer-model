"""Average checkpoints, tune decoding on dev, write dev/test predictions.

    python -m src.predict --config configs/base.yaml --avg 5 --sweep
    python -m src.predict --ckpt checkpoints/base/best.pt --beam 5 --alpha 1.0

Selection uses dev only (PLAN.md D4). Writes into <out_dir>/eval/:
  dev_predictions.json  dev_report.json  sweep.json  test_predictions.json
"""
import argparse
import glob
import json
import os
import time

import sentencepiece as spm
import torch

from .decode import Translator
from .metrics import score_predictions
from .model import Seq2SeqTransformer
from .utils import get_device, load_config, read_jsonl, write_json


def load_model(paths, device):
    """Load one checkpoint, or the parameter-wise mean of several (checkpoint averaging)."""
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    model = Seq2SeqTransformer(**cks[0]["model_config"])
    if len(cks) == 1:
        model.load_state_dict(cks[0]["model"])
    else:
        avg = {k: sum(c["model"][k].float() for c in cks) / len(cks) for k in cks[0]["model"]}
        model.load_state_dict(avg)
    return model.to(device).eval()


def run(tr, rows):
    return {r["id"]: h for r, h in zip(rows, tr.translate([r["source"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out_dir")
    ap.add_argument("--ckpt", nargs="*", help="explicit checkpoint file(s); default: last --avg step ckpts")
    ap.add_argument("--avg", type=int, default=5)
    ap.add_argument("--beam", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--sweep", action="store_true", help="grid over beam x length penalty on dev")
    ap.add_argument("--no_split", action="store_true", help="ablation: never split long inputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out_dir or cfg["out_dir"]
    dc = cfg["decode"]
    device = get_device()
    sp = spm.SentencePieceProcessor(model_file=f"{out}/spm.model")
    paths = args.ckpt or sorted(glob.glob(f"{out}/step_*.pt"))[-args.avg:]
    print("checkpoints:", [os.path.basename(p) for p in paths])
    model = load_model(paths, device)

    dev_in, dev_gold = read_jsonl("data/dev/inputs.jsonl"), read_jsonl("data/dev/labels.jsonl")
    test_in = read_jsonl("data/test/inputs.jsonl")
    split = 10 ** 9 if args.no_split else dc["split_over_tokens"]
    make = lambda beam, alpha: Translator(model, sp, device, beam=beam, len_penalty=alpha,
                                          max_len_a=dc["max_len_a"], max_len_b=dc["max_len_b"],
                                          split_over_tokens=split)
    os.makedirs(f"{out}/eval", exist_ok=True)

    beam, alpha = args.beam or dc["beam"], args.alpha if args.alpha is not None else dc["len_penalty"]
    if args.sweep:
        results = []
        for b in [1, 4, 5, 8]:
            for a in ([0.0] if b == 1 else [0.6, 1.0, 1.4]):
                t0 = time.time()
                rep = score_predictions(run(make(b, a), dev_in), dev_gold)
                results.append({"beam": b, "alpha": a, "overall": rep["OVERALL"], "bleu": rep["all"]["bleu"],
                                "chrf": rep["all"]["chrf"], "by_slice": rep["by_slice"], "sec": time.time() - t0})
                print(f"beam={b} alpha={a}: OVERALL {rep['OVERALL']:.2f} BLEU {rep['all']['bleu']:.2f} "
                      f"chrF {rep['all']['chrf']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        write_json(results, f"{out}/eval/sweep.json")
        best = max(results, key=lambda r: r["overall"])
        beam, alpha = best["beam"], best["alpha"]
        print(f"selected beam={beam} alpha={alpha}")

    tr = make(beam, alpha)
    dev_pred = run(tr, dev_in)
    rep = score_predictions(dev_pred, dev_gold)
    rep["decode"] = {"beam": beam, "alpha": alpha, "split": not args.no_split, "checkpoints": paths}
    write_json(dev_pred, f"{out}/eval/dev_predictions.json")
    write_json(rep, f"{out}/eval/dev_report.json")
    print(json.dumps({k: rep[k] for k in ["OVERALL", "all", "by_slice"]}, indent=1))

    test_pred = run(tr, test_in)
    assert set(test_pred) == {r["id"] for r in test_in}
    write_json(test_pred, f"{out}/eval/test_predictions.json")
    print(f"wrote {len(test_pred)} test predictions")


if __name__ == "__main__":
    main()
