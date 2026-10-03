"""Fast correctness checks for the model and decoding (CPU, < 1 min).

    python -m pytest -q tests/   (or: python tests/test_sanity.py)
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.decode import beam_search, greedy, split_long  # noqa: E402
from src.model import BOS, EOS, PAD, Seq2SeqTransformer  # noqa: E402
from src.utils import normalize  # noqa: E402


def tiny(pos="rope"):
    torch.manual_seed(0)
    return Seq2SeqTransformer(50, d_model=32, n_heads=4, d_ff=64, enc_layers=2, dec_layers=2,
                              dropout=0.0, pos=pos, max_len=64).eval()


def test_causal_decoder():
    """Changing a later target token must not change logits at earlier positions."""
    m = tiny()
    src = torch.randint(4, 50, (2, 7))
    tgt = torch.randint(4, 50, (2, 6))
    a = m(src, tgt)
    tgt2 = tgt.clone()
    tgt2[:, 4:] = 5
    b = m(src, tgt2)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_source_padding_ignored():
    """Appending PAD to the source must not change the output."""
    for pos in ["rope", "sinusoidal"]:
        m = tiny(pos)
        src = torch.randint(4, 50, (1, 5))
        tgt = torch.randint(4, 50, (1, 4))
        padded = torch.cat([src, torch.full((1, 3), PAD)], 1)
        assert torch.allclose(m(src, tgt), m(padded, tgt), atol=1e-5)


def test_overfit_and_decode():
    """A tiny model memorises 8 pairs; greedy == beam(1) and beam(4) recovers the targets."""
    torch.manual_seed(0)
    m = tiny().train()
    src = torch.randint(4, 50, (8, 6))
    tgt_body = torch.randint(4, 50, (8, 5))
    src[:, -1] = EOS
    tgt = torch.cat([torch.full((8, 1), BOS), tgt_body, torch.full((8, 1), EOS)], 1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(400):
        loss = F.cross_entropy(m(src, tgt[:, :-1]).transpose(1, 2), tgt[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()
    m.eval()
    g = greedy(m, src)
    assert g == tgt_body.tolist()
    assert beam_search(m, src, beam=1, alpha=0.0) == g
    assert beam_search(m, src, beam=4, alpha=1.0) == tgt_body.tolist()


def test_normalize():
    assert normalize("Ne t’inquiète pas !") == "Ne t'inquiète pas !"
    assert normalize("« Bonjour », dit-il") == '"Bonjour", dit-il'
    assert normalize("Don’t “go”…") == "Don't \"go\"..."


def test_split_long():
    assert split_long("17 Il dit. 18 Puis il partit! Fin") == ["17 Il dit.", "18 Puis il partit!", "Fin"]
    assert split_long("M. Dupont est venu.") == ["M. Dupont est venu."]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
