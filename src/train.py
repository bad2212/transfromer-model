"""Train the translation model. Resumable: re-running the same command continues from
<out_dir>/last.pt (Colab disconnects), including the W&B run and the data position.

    python -m src.train --config configs/base.yaml [--out_dir /content/drive/MyDrive/ckpt]
"""
import argparse
import glob
import math
import os
import random
import shutil
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .decode import Translator
from .metrics import flat, score_predictions
from .model import BOS, EOS, PAD, Seq2SeqTransformer
from .utils import get_device, load_config, read_jsonl, seed_everything


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def read_lines(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


class PairDataset(Dataset):
    """Encodes on the fly so the source side can use SentencePiece sampling
    (subword regularization, Kudo 2018). Target segmentation stays deterministic."""

    def __init__(self, src, tgt, spm_path, alpha, max_tokens):
        self.src, self.tgt, self.spm_path = src, tgt, spm_path
        self.alpha, self.max_tokens = alpha, max_tokens
        self.sp = None

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        if self.sp is None:                      # lazily, once per DataLoader worker
            self.sp = spm.SentencePieceProcessor(model_file=self.spm_path)
        if self.alpha > 0:
            s = self.sp.encode(self.src[i], enable_sampling=True, alpha=self.alpha, nbest_size=-1)
        else:
            s = self.sp.encode(self.src[i])
        t = self.sp.encode(self.tgt[i])
        return s[: self.max_tokens] + [EOS], [BOS] + t[: self.max_tokens] + [EOS]


def collate(batch):
    S = max(len(s) for s, _ in batch)
    T = max(len(t) for _, t in batch)
    src = torch.full((len(batch), S), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), T), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)], tgt[i, :len(t)] = torch.tensor(s), torch.tensor(t)
    return src, tgt


def make_batches(lengths, batch_tokens, seed, epoch):
    """Token-budget batches of similar length. Deterministic per (seed, epoch) so a resumed
    run can skip exactly the batches it already consumed."""
    rng = random.Random(seed * 1000 + epoch)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    batches, chunk = [], 100_000                 # sort within large random chunks
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: lengths[i])
        cur, cur_max = [], 0
        for i in part:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > batch_tokens:
                batches.append(cur)
                cur, m = [], lengths[i]
            cur.append(i)
            cur_max = m
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def lr_at(step, peak, warmup):
    """Linear warmup then inverse-sqrt decay (Vaswani et al. 2017)."""
    step = max(step, 1)
    return peak * min(step / warmup, math.sqrt(warmup / step))


@torch.no_grad()
def evaluate(model, sp, device, cfg, valid_pairs, dev_inputs, dev_gold):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(valid_pairs), 64):
        chunk = valid_pairs[i:i + 64]
        src, tgt = collate([(sp.encode(s)[:255] + [EOS], [BOS] + sp.encode(t)[:255] + [EOS]) for s, t in chunk])
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.float().transpose(1, 2), tgt[:, 1:], ignore_index=PAD, reduction="sum")
        tot_loss += loss.item()
        tot_tok += (tgt[:, 1:] != PAD).sum().item()
    dc = cfg["decode"]
    tr = Translator(model, sp, device, beam=1, max_len_a=dc["max_len_a"], max_len_b=dc["max_len_b"],
                    split_over_tokens=dc["split_over_tokens"])
    hyps = tr.translate([r["source"] for r in dev_inputs])
    report = score_predictions({r["id"]: h for r, h in zip(dev_inputs, hyps)}, dev_gold)
    model.train()
    return tot_loss / tot_tok, report, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", help="override cfg.out_dir (e.g. a Google Drive path)")
    ap.add_argument("--max_steps", type=int, help="override (used to test resume)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    load_env()
    seed_everything(cfg["seed"])
    device = get_device()
    tc, work, out = cfg["train"], cfg["work_dir"], cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    shutil.copy(f"{work}/spm.model", f"{out}/spm.model")

    sp = spm.SentencePieceProcessor(model_file=f"{work}/spm.model")
    src_lines, tgt_lines = read_lines(f"{work}/train.fr"), read_lines(f"{work}/train.en")
    max_tok = cfg["data"]["max_tokens"]
    t0 = time.time()
    s_len, t_len = sp.encode(src_lines), sp.encode(tgt_lines)
    keep = [i for i in range(len(src_lines)) if len(s_len[i]) <= max_tok and len(t_len[i]) <= max_tok]
    lengths = [max(len(s_len[i]), len(t_len[i])) + 2 for i in keep]
    src_lines, tgt_lines = [src_lines[i] for i in keep], [tgt_lines[i] for i in keep]
    del s_len, t_len
    print(f"train pairs {len(keep)} (len filter <= {max_tok}) tokenized in {time.time() - t0:.0f}s")

    ds = PairDataset(src_lines, tgt_lines, f"{work}/spm.model", cfg["tokenizer"]["src_sampling_alpha"], max_tok)
    valid_pairs = list(zip(read_lines(f"{work}/valid.fr"), read_lines(f"{work}/valid.en")))
    dev_inputs, dev_gold = read_jsonl("data/dev/inputs.jsonl"), read_jsonl("data/dev/labels.jsonl")

    model = Seq2SeqTransformer(sp.get_piece_size(), **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params {n_params / 1e6:.1f}M on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "train_seconds": 0.0, "best": -1.0, "wandb_id": None}
    if os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        torch.set_rng_state(ck["rng_cpu"].cpu())  # map_location moved it to the GPU
        print(f"resumed from step {state['step']} (epoch {state['epoch']}, batch {state['batch_in_epoch']})")

    run = None
    if cfg["wandb"]["enabled"]:
        import wandb
        state["wandb_id"] = state["wandb_id"] or wandb.util.generate_id()
        run = wandb.init(project=cfg["wandb"]["project"], id=state["wandb_id"], resume="allow",
                         config={**cfg, "n_params": n_params, "train_pairs": len(keep)},
                         name=os.path.basename(out.rstrip("/")))

    def save(path_extra=None):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
              "state": state, "rng_cpu": torch.get_rng_state(), "config": cfg, "model_config": model.config}
        torch.save(ck, f"{out}/last.pt.tmp")
        os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")     # atomic: a disconnect never corrupts it
        if path_extra:
            torch.save({"model": model.state_dict(), "model_config": model.config, "step": state["step"]}, path_extra)

    model.train()
    budget = tc["max_hours"] * 3600
    tick, tok_count, loss_sum, loss_n = time.time(), 0, 0.0, 0
    stop = False
    while not stop:
        batches = make_batches(lengths, tc["batch_tokens"], cfg["seed"], state["epoch"])
        g = torch.Generator()
        g.manual_seed(cfg["seed"] + state["epoch"])
        loader = DataLoader(ds, batch_sampler=batches[state["batch_in_epoch"]:], collate_fn=collate,
                            num_workers=tc["num_workers"], generator=g, persistent_workers=False)
        micro = 0
        for src, tgt in loader:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(src, tgt[:, :-1])
                loss = F.cross_entropy(logits.float().transpose(1, 2), tgt[:, 1:], ignore_index=PAD,
                                       label_smoothing=tc["label_smoothing"])
            scaler.scale(loss / tc["accum_steps"]).backward()
            state["batch_in_epoch"] += 1
            micro += 1
            tok_count += (tgt[:, 1:] != PAD).sum()       # stays on device: no per-batch sync
            loss_sum, loss_n = loss_sum + loss.detach(), loss_n + 1
            if micro % tc["accum_steps"]:
                continue

            lr = lr_at(state["step"] + 1, tc["lr"], tc["warmup_steps"])
            for group in opt.param_groups:
                group["lr"] = lr
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            step = state["step"]

            if step % 100 == 0:
                now = time.time()
                state["train_seconds"] += now - tick
                msg = {"train/loss": (loss_sum / loss_n).item(), "train/lr": lr, "train/grad_norm": gnorm.item(),
                       "train/tgt_tokens_per_s": float(tok_count) / (now - tick), "train/epoch": state["epoch"],
                       "train/hours": state["train_seconds"] / 3600}
                print(f"step {step} " + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in msg.items()), flush=True)
                if run:
                    run.log(msg, step=step)
                tick, tok_count, loss_sum, loss_n = now, 0, 0.0, 0

            if step % tc["eval_every"] == 0:
                vloss, report, hyps = evaluate(model, sp, device, cfg, valid_pairs, dev_inputs, dev_gold)
                logs = {"val/loss": vloss, "val/ppl": math.exp(vloss), **flat(report)}
                print(f"eval step {step}: val_loss={vloss:.3f} dev OVERALL={report['OVERALL']:.2f} "
                      + " ".join(f"{k}={v['bleu']:.1f}/{v['chrf']:.1f}" for k, v in report["by_slice"].items()),
                      flush=True)
                if run:
                    import wandb
                    logs["dev/samples"] = wandb.Table(columns=["id", "slice", "source", "hyp", "ref"], data=[
                        [r["id"], r["slice"], r["source"], h, g_["reference"]]
                        for r, h, g_ in list(zip(dev_inputs, hyps, dev_gold))[:20]])
                    run.log(logs, step=step)
                if report["OVERALL"] > state["best"]:
                    state["best"] = report["OVERALL"]
                    torch.save({"model": model.state_dict(), "model_config": model.config, "step": step},
                               f"{out}/best.pt")
                tick = time.time()               # don't count eval time as training throughput

            if step % tc["ckpt_every"] == 0:
                save(f"{out}/step_{step:07d}.pt")
                for old in sorted(glob.glob(f"{out}/step_*.pt"))[: -tc["keep_ckpts"]]:
                    os.remove(old)

            if step >= tc["max_steps"] or state["train_seconds"] + (time.time() - tick) > budget:
                stop = True
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0

    state["train_seconds"] += time.time() - tick
    save(f"{out}/step_{state['step']:07d}.pt")
    print(f"done at step {state['step']}, {state['train_seconds'] / 3600:.2f} h, best dev OVERALL {state['best']:.2f}")
    if run:
        run.summary["best_dev_overall"] = state["best"]
        run.finish()


if __name__ == "__main__":
    main()
