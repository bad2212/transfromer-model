#!/usr/bin/env python3
"""Official scorer for the French->English translation challenge.

    python3 score.py --gold data/dev/labels.jsonl --pred my_dev_predictions.json

`--gold` is a JSONL file, one object per line:
    {"id": "...", "reference": "<english>", "slice": "seen|long|unseen_domain"}
`--pred` is ONE JSON file: {"<id>": "<english translation>", ...}  (see sample_submission.json).
An id missing from the file scores as an empty translation.

Metrics (overall and per slice), both lowercased first:
  BLEU   corpus BLEU, 4-gram, +1 add-one smoothing on higher-order n-grams, brevity penalty.
  chrF   character n-gram F-score (n=1..6, beta=2), sentence-averaged.

OVERALL = 0.40 * BLEU(all) + 0.40 * chrF(all) + 0.20 * chrF(unseen_domain slice)

NOTE: single-reference, whitespace/punct tokenization. Treat as relative, not identical to
sacreBLEU. Standard library only.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict


def wtok(s):
    return re.findall(r"\w+|[^\w\s]", (s or "").lower(), flags=re.UNICODE)


def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)) if len(toks) >= n else Counter()


def corpus_bleu(hyps, refs, max_n=4):
    """Corpus BLEU with add-one smoothing on n>=2 (Chen & Cherry method 1-ish)."""
    match = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    hyp_len = ref_len = 0
    for h, r in zip(hyps, refs):
        ht, rt = wtok(h), wtok(r)
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            hn, rn = ngrams(ht, n), ngrams(rt, n)
            overlap = sum(min(c, rn[g]) for g, c in hn.items())
            match[n] += overlap
            total[n] += max(len(ht) - n + 1, 0)
    if hyp_len == 0:
        return 0.0
    precs = []
    for n in range(1, max_n + 1):
        m, t = match[n], total[n]
        if t == 0:
            precs.append(0.0)
        elif n == 1:
            precs.append(m / t if m > 0 else 1e-9)
        else:
            precs.append((m + 1) / (t + 1))  # smoothing
    if min(precs) <= 0:
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precs) / max_n)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * geo


def chrf_sentence(h, r, max_n=6, beta=2.0):
    h = re.sub(r"\s+", "", (h or "").lower())
    r = re.sub(r"\s+", "", (r or "").lower())
    if not h and not r:
        return 100.0
    if not h or not r:
        return 0.0
    f_scores = []
    for n in range(1, max_n + 1):
        hn = ngrams(list(h), n)
        rn = ngrams(list(r), n)
        if not hn or not rn:
            continue
        overlap = sum(min(c, rn[g]) for g, c in hn.items())
        p = overlap / max(sum(hn.values()), 1)
        rec = overlap / max(sum(rn.values()), 1)
        if p + rec == 0:
            f_scores.append(0.0)
        else:
            b2 = beta * beta
            f_scores.append((1 + b2) * p * rec / (b2 * p + rec))
    return 100.0 * (sum(f_scores) / len(f_scores)) if f_scores else 0.0


def score_slice(hyps, refs):
    if not refs:
        return None
    bleu = corpus_bleu(hyps, refs)
    chrf = sum(chrf_sentence(h, r) for h, r in zip(hyps, refs)) / len(refs)
    return {"n": len(refs), "bleu": bleu, "chrf": chrf}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gold", required=True, help="gold labels JSONL")
    ap.add_argument("--pred", required=True, help="submission JSON file")
    ap.add_argument("--out", help="optional: write the full report here as JSON")
    a = ap.parse_args()

    gold = {}
    with open(a.gold) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                gold[o["id"]] = o
    if not gold:
        sys.exit(f"no gold labels found in {a.gold}")
    pred = json.load(open(a.pred))
    if not isinstance(pred, dict):
        sys.exit("submission must be a JSON object keyed by id")
    unknown = set(pred) - set(gold)
    missing = sum(1 for d in gold if d not in pred)

    ids = list(gold)
    hyps = [pred.get(i, "") for i in ids]
    refs = [gold[i].get("reference", "") for i in ids]
    by = defaultdict(lambda: ([], []))
    for i in ids:
        h, r = pred.get(i, ""), gold[i].get("reference", "")
        sl = gold[i].get("slice", "unspecified")
        by[sl][0].append(h)
        by[sl][1].append(r)

    A = score_slice(hyps, refs)
    sl = {k: score_slice(h, r) for k, (h, r) in by.items()}
    ud = sl.get("unseen_domain", {}).get("chrf", A["chrf"]) if sl.get("unseen_domain") else A["chrf"]
    overall = 0.40 * A["bleu"] + 0.40 * A["chrf"] + 0.20 * ud

    print(f"sentences scored        {A['n']}   (missing: {missing}, unknown ids ignored: {len(unknown)})")
    print(f"BLEU  (all)             {A['bleu']:6.2f}")
    print(f"chrF  (all)             {A['chrf']:6.2f}")
    print(f"OVERALL                 {overall:6.2f}")
    print("by slice:")
    for k in ["seen", "long", "unseen_domain"]:
        v = sl.get(k)
        if v:
            print(f"   {k:<16} BLEU {v['bleu']:6.2f}   chrF {v['chrf']:6.2f}   (n={v['n']})")

    if a.out:
        json.dump({"all": A, "OVERALL": overall, "by_slice": sl}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
