"""Pre-LN encoder-decoder Transformer with RoPE (or sinusoidal) positions.

Design (see PLAN.md / report):
  * encoder-decoder (Vaswani et al. 2017), pre-LayerNorm (Xiong et al. 2020)
  * rotary position embeddings in self-attention (Su et al. 2021); cross-attention
    gets no positions (it aligns by content). `pos: sinusoidal` is kept for ablation.
  * one embedding matrix shared by encoder input, decoder input and output
    projection (Press & Wolf 2017) - possible because the vocab is joint fr+en.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class Rotary(nn.Module):
    """Precomputed cos/sin tables; rotates pairs of channels by a position-dependent angle."""

    def __init__(self, head_dim, max_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.outer(torch.arange(max_len).float(), inv_freq)   # (T, hd/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):                       # x: (B, H, T, hd)
        T = x.size(-2)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, h, dropout, rotary=None):
        super().__init__()
        self.h, self.hd = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.rotary = rotary
        self.dropout = dropout

    def split(self, x):                         # (B, T, d) -> (B, H, T, hd)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.hd).transpose(1, 2)

    def forward(self, x, kv, mask=None, causal=False):
        q, k, v = self.split(self.q(x)), self.split(self.k(kv)), self.split(self.v(kv))
        if self.rotary is not None:             # self-attention only
            q, k = self.rotary(q), self.rotary(k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0)
        B, H, T, hd = out.shape
        return self.o(out.transpose(1, 2).reshape(B, T, H * hd))


class FeedForward(nn.Module):
    def __init__(self, d, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout, rotary):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, h, dropout, rotary)
        self.ff = FeedForward(d, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        y = self.ln1(x)
        x = x + self.drop(self.attn(y, y, mask=src_mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, dropout, rotary):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, h, dropout, rotary)
        self.cross_attn = Attention(d, h, dropout, rotary=None)
        self.ff = FeedForward(d, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask):
        # Causal mask alone is enough: target padding sits after real tokens, so real
        # positions never see it, and padded positions are ignored by the loss.
        y = self.ln1(x)
        x = x + self.drop(self.self_attn(y, y, causal=True))
        x = x + self.drop(self.cross_attn(self.ln2(x), memory, mask=src_mask))
        return x + self.drop(self.ff(self.ln3(x)))


def sinusoidal_table(max_len, d):
    pos = torch.arange(max_len).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, enc_layers=6,
                 dec_layers=6, dropout=0.1, pos="rope", max_len=512):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                           enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout,
                           pos=pos, max_len=max_len)
        self.d = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        rotary = Rotary(d_model // n_heads, max_len) if pos == "rope" else None
        if pos == "sinusoidal":
            self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)
        self.pos = pos
        self.encoder = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout, rotary) for _ in range(enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout, rotary) for _ in range(dec_layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embed.weight, std=self.d ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    def _embed(self, ids):
        x = self.embed(ids) * math.sqrt(self.d)
        if self.pos == "sinusoidal":
            x = x + self.pe[: ids.size(1)]
        return self.drop(x)

    @staticmethod
    def src_mask(src):
        """(B, 1, 1, S) boolean, True = may attend (SDPA convention)."""
        return (src != PAD)[:, None, None, :]

    def encode(self, src):
        mask = self.src_mask(src)
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_ln(x), mask

    def decode(self, tgt_in, memory, src_mask):
        x = self._embed(tgt_in)
        for layer in self.decoder:
            x = layer(x, memory, src_mask)
        return F.linear(self.dec_ln(x), self.embed.weight)   # tied output projection

    def forward(self, src, tgt_in):
        memory, mask = self.encode(src)
        return self.decode(tgt_in, memory, mask)
