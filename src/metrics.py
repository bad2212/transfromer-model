"""Dev scoring through the official score.py functions (no re-implementation)."""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score as official  # noqa: E402  (repo-root score.py, kept untouched)


def score_predictions(preds, gold_rows):
    """preds: {id: hyp}; gold_rows: dicts with id/reference/slice. Mirrors score.py main()."""
    hyps = [preds.get(g["id"], "") for g in gold_rows]
    refs = [g["reference"] for g in gold_rows]
    by = defaultdict(lambda: ([], []))
    for g, h in zip(gold_rows, hyps):
        by[g.get("slice", "unspecified")][0].append(h)
        by[g.get("slice", "unspecified")][1].append(g["reference"])
    A = official.score_slice(hyps, refs)
    slices = {k: official.score_slice(h, r) for k, (h, r) in by.items()}
    ud = slices["unseen_domain"]["chrf"] if "unseen_domain" in slices else A["chrf"]
    overall = 0.40 * A["bleu"] + 0.40 * A["chrf"] + 0.20 * ud
    return {"all": A, "OVERALL": overall, "by_slice": slices}


def flat(report, prefix="dev"):
    """Flatten for W&B logging."""
    out = {f"{prefix}/overall": report["OVERALL"], f"{prefix}/bleu": report["all"]["bleu"],
           f"{prefix}/chrf": report["all"]["chrf"]}
    for k, v in report["by_slice"].items():
        out[f"{prefix}/{k}/bleu"], out[f"{prefix}/{k}/chrf"] = v["bleu"], v["chrf"]
    return out
