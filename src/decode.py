"""Greedy and beam-search decoding plus a text-in/text-out Translator.

No KV cache by design (PLAN.md D3): each step re-runs the decoder over the prefix.
For ~500 eval sentences that costs seconds and keeps the search easy to verify.
"""
import re

import torch
import torch.nn.functional as F

from .model import BOS, EOS, PAD
from .utils import normalize


def _max_lens(src, a, b, cap):
    src_len = (src != PAD).sum(1).float()
    return (src_len * a + b).long().clamp(max=cap)


@torch.no_grad()
def greedy(model, src, max_len_a=1.5, max_len_b=10):
    memory, mask = model.encode(src)
    max_lens = _max_lens(src, max_len_a, max_len_b, model.config["max_len"] - 1)
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for step in range(int(max_lens.max())):
        nxt = model.decode(ys, memory, mask)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= (nxt == EOS) | (step + 1 >= max_lens)
        if done.all():
            break
    return [_strip(row) for row in ys[:, 1:].tolist()]


def _strip(ids):
    out = []
    for t in ids:
        if t in (EOS, PAD):
            break
        out.append(t)
    return out


def _lp(length, alpha):
    """GNMT length penalty (Wu et al. 2016): larger alpha favours longer outputs."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(model, src, beam=5, alpha=1.0, max_len_a=1.5, max_len_b=10):
    B, dev = src.size(0), src.device
    memory, mask = model.encode(src)
    memory, mask = memory.repeat_interleave(beam, 0), mask.repeat_interleave(beam, 0)
    max_lens = _max_lens(src, max_len_a, max_len_b, model.config["max_len"] - 1).tolist()

    seqs = torch.full((B * beam, 1), BOS, dtype=torch.long, device=dev)
    scores = torch.zeros(B, beam, device=dev)
    scores[:, 1:] = float("-inf")             # all beams start identical: keep only one
    finished = [[] for _ in range(B)]          # per sentence: (normalised score, token ids)
    active = [True] * B

    for step in range(max(max_lens)):
        logp = F.log_softmax(model.decode(seqs, memory, mask)[:, -1].float(), -1)
        V = logp.size(-1)
        cand = (scores.view(-1, 1) + logp).view(B, beam * V)
        top_s, top_i = cand.topk(2 * beam, dim=1)      # 2k so k survive after EOS removal
        top_s, top_i = top_s.tolist(), top_i.tolist()

        new_rows, new_scores = [], []
        for b in range(B):
            rows, sc = [], []
            if active[b]:
                for s, idx in zip(top_s[b], top_i[b]):
                    if s == float("-inf"):
                        break
                    src_beam, tok = divmod(idx, V)
                    row = b * beam + src_beam
                    if tok == EOS:
                        finished[b].append((s / _lp(step + 1, alpha), seqs[row, 1:].tolist()))
                    else:
                        rows.append((row, tok))
                        sc.append(s)
                    if len(rows) == beam:
                        break
                last_step = step + 1 >= max_lens[b]
                if last_step:                   # out of length budget: finalise live beams
                    for (row, tok), s in zip(rows, sc):
                        finished[b].append((s / _lp(step + 1, alpha), seqs[row, 1:].tolist() + [tok]))
                if len(finished[b]) >= beam or last_step or not rows:
                    active[b] = False
            if not active[b]:                   # keep shapes fixed with dead placeholder beams
                rows, sc = [], []
            rows += [(b * beam, PAD)] * (beam - len(rows))
            sc += [float("-inf")] * (beam - len(sc))
            new_rows.extend(rows)
            new_scores.append(sc)

        if not any(active):
            break
        idx = torch.tensor([r for r, _ in new_rows], device=dev)
        toks = torch.tensor([t for _, t in new_rows], device=dev)
        seqs = torch.cat([seqs[idx], toks[:, None]], 1)
        scores = torch.tensor(new_scores, device=dev)

    return [max(f)[1] if f else [] for f in finished]


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[\"'(\[]?[A-ZÀ-Ý0-9])")


_ABBREV = re.compile(r"(?:^|\s)(?:[A-Z]|M|Mme|Mlle|MM|Dr|St|Ste|art|p|no|n°|cf|etc)\.$", re.IGNORECASE)


def split_long(text):
    """Split on sentence-final punctuation followed by a capital/digit (e.g. verse numbers),
    but never right after an abbreviation such as "M." or "Dr."."""
    pieces = []
    for p in _SENT_SPLIT.split(text):
        if pieces and _ABBREV.search(pieces[-1]):
            pieces[-1] += " " + p
        elif p.strip():
            pieces.append(p)
    return pieces


class Translator:
    def __init__(self, model, sp, device, beam=5, len_penalty=1.0, max_len_a=1.5,
                 max_len_b=10, split_over_tokens=128, batch_size=64, batch_tokens=2048):
        self.model, self.sp, self.device = model.eval(), sp, device
        self.beam, self.alpha = beam, len_penalty
        self.a, self.b = max_len_a, max_len_b
        self.split_over = split_over_tokens
        self.batch_size, self.batch_tokens = batch_size, batch_tokens

    def _encode(self, text):
        return self.sp.encode(text)[: self.model.config["max_len"] - 1] + [EOS]

    def _batches(self, segments):
        """Length-sorted batches capped by a source-token budget, so one very long input doesn't
        drag 60 short ones (x beam) through hundreds of extra decoding steps."""
        order = sorted(range(len(segments)), key=lambda i: len(segments[i]))
        batch = []
        for i in order:
            if batch and (len(batch) == self.batch_size or len(segments[i]) * (len(batch) + 1) > self.batch_tokens):
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def _translate_segments(self, segments):
        out = [None] * len(segments)
        for ids in self._batches(segments):
            seqs = [segments[i] for i in ids]
            src = torch.full((len(seqs), max(map(len, seqs))), PAD, dtype=torch.long)
            for r, s in enumerate(seqs):
                src[r, :len(s)] = torch.tensor(s)
            src = src.to(self.device)
            if self.beam > 1:
                hyps = beam_search(self.model, src, self.beam, self.alpha, self.a, self.b)
            else:
                hyps = greedy(self.model, src, self.a, self.b)
            for i, h in zip(ids, hyps):
                out[i] = normalize(self.sp.decode(h))
            if self.device.type == "mps":
                torch.mps.empty_cache()             # release cached GPU buffers between batches
        return out

    def translate(self, texts):
        """Normalise -> (split if very long) -> decode -> rejoin -> normalise output."""
        segments, owner = [], []
        for i, t in enumerate(texts):
            t = normalize(t)
            pieces = split_long(t) if len(self.sp.encode(t)) > self.split_over else [t]
            for p in pieces:
                segments.append(self._encode(p))
                owner.append(i)
        outs = self._translate_segments(segments)
        joined = [[] for _ in texts]
        for i, o in zip(owner, outs):
            joined[i].append(o)
        return [" ".join(parts) for parts in joined]
