"""Build report/report.pdf (3 pages) and report/appendix.pdf from the result files, so every number
in them is traceable.

    python report/build_report.py --run checkpoints/base_6x3 [--chrome /path/to/chrome]

Reads: <run>/eval/dev_report.json, <run>/eval/sweep.json, <run>/analysis/analysis.json,
       <run>/analysis/examples.md, work/data_stats.json, report/validations.py.
Writes report/{report,appendix}.{html,pdf}; PDFs are printed with headless Chrome.
"""
import argparse
import base64
import html
import io
import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LINKS = [
    ("GitHub", "bad2212/transfromer-model", "https://github.com/bad2212/transfromer-model"),
    ("Hugging Face", "Badalt/fr-en-transformer-scratch", "https://huggingface.co/Badalt/fr-en-transformer-scratch"),
    ("W&B run", "5ks4crpz", "https://wandb.ai/badalthakur2212-iisc/fr-en-transformer/runs/5ks4crpz"),
    ("W&B report", "dashboards",
     "https://wandb.ai/badalthakur2212-iisc/fr-en-transformer/reports/FR%E2%86%92EN-Transformer-from-scratch:-training-and-generalization--VmlldzoxODA1MDg0Mw=="),
]
# Greedy dev evaluations logged during training (W&B run 5ks4crpz), single checkpoint.
CURVE = [(2000, 4.11, 14.45), (4000, 2.65, 30.79), (6000, 2.28, 35.39), (8000, 2.14, 37.58)]

# Two-series palette (validated: CVD dE 24.7, normal-vision dE 33.6, >= 3:1 on white).
IN_DOMAIN, BOOKS = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#1b1d21", "#5b616b", "#e3e6ea"

CSS = """
@page { size: A4; margin: 12mm 14mm 12mm 14mm; }
* { box-sizing: border-box; }
body { font: 8.7pt/1.45 -apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif; color: #1b1d21;
       margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; font-variant-numeric: tabular-nums; }
a { color: #1f63c4; text-decoration: none; }
p { margin: 0 0 5px; } ul, ol { margin: 0 0 5px; padding-left: 16px; } li { margin: 0 0 2px; }
b, strong { font-weight: 600; } i { color: #2c3036; }
code { font: 7.8pt 'SF Mono', Menlo, monospace; background: #f1f3f5; padding: 0 3px; border-radius: 3px; }

.title { border-bottom: 2px solid #1b1d21; padding-bottom: 6px; margin-bottom: 8px; }
.eyebrow { font-size: 7.2pt; letter-spacing: .08em; text-transform: uppercase; color: #2a78d6; font-weight: 600; }
h1 { font-size: 16pt; line-height: 1.2; margin: 2px 0 3px; letter-spacing: -.01em; }
.sub { color: #5b616b; font-size: 8.4pt; margin: 0 0 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip { font-size: 7.4pt; border: 1px solid #d5dae0; border-radius: 10px; padding: 1px 8px; color: #1b1d21; white-space: nowrap; }
.chip b { color: #5b616b; font-weight: 600; margin-right: 3px; }

.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin: 8px 0 8px; }
.kpi { background: #f5f7fa; border-radius: 6px; padding: 6px 9px 5px; }
.kpi .v { font-size: 15pt; font-weight: 650; line-height: 1.1; }
.kpi .v small { font-size: 8pt; font-weight: 500; color: #5b616b; }
.kpi .l { font-size: 7.3pt; color: #5b616b; margin-top: 2px; white-space: nowrap; }
.kpi.accent { background: #eaf2fc; } .kpi.accent .v { color: #1f63c4; }
.kpi.warn .v { color: #c4501f; }

.callout { background: #f3f7fd; border-left: 3px solid #2a78d6; border-radius: 0 6px 6px 0; padding: 7px 11px; margin: 4px 0 6px; }
.callout ol { margin: 0; padding-left: 15px; } .callout li { margin: 0 0 2px; }
.callout .h { font-size: 7.2pt; letter-spacing: .07em; text-transform: uppercase; color: #1f63c4; font-weight: 650; margin-bottom: 3px; }

h2 { font-size: 10.6pt; margin: 13px 0 5px; display: flex; align-items: center; gap: 7px; break-after: avoid; }
h2 .n { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 4px;
        background: #1b1d21; color: #fff; font-size: 7.6pt; font-weight: 650; }
h3 { font-size: 8.8pt; margin: 8px 0 3px; break-after: avoid; }

table { border-collapse: collapse; width: 100%; margin: 3px 0 7px; font-size: 7.9pt; break-inside: avoid; }
th { text-align: left; font-size: 7.2pt; font-weight: 650; color: #5b616b;
     border-bottom: 1.3px solid #1b1d21; padding: 3px 6px; vertical-align: bottom; }
td { border-bottom: 1px solid #e3e6ea; padding: 3px 6px; vertical-align: top; }
table.long { break-inside: auto; } tr { break-inside: avoid; }
tr.total td { border-top: 1.3px solid #1b1d21; border-bottom: none; font-weight: 650; }
tr.hl td { background: #f3f7fd; }
td.n, th.n { text-align: right; white-space: nowrap; }
td.ci { color: #5b616b; text-align: right; white-space: nowrap; }
td.k { font-weight: 600; white-space: nowrap; }
.muted { color: #5b616b; }

.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.grid2 > div { min-width: 0; }
figure { margin: 2px 0 6px; break-inside: avoid; }
figure .ft { font-size: 8.4pt; font-weight: 650; margin-bottom: 1px; }
figure .fs { font-size: 7.2pt; color: #5b616b; margin-bottom: 2px; }
figure img { width: 100%; display: block; }
.key { display: flex; gap: 12px; font-size: 7.3pt; color: #5b616b; margin: 0 0 2px; }
.key span::before { content: ''; display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.key .a::before { background: #2a78d6; } .key .b::before { background: #eb6834; }

.finding { display: grid; grid-template-columns: 18px 1fr; gap: 0 6px; margin: 0 0 5px; break-inside: avoid; }
.finding .i { width: 16px; height: 16px; border-radius: 50%; background: #eaf2fc; color: #1f63c4; font-size: 7.6pt; font-weight: 650;
              display: flex; align-items: center; justify-content: center; margin-top: 1px; }
.refs { font-size: 7.1pt; color: #3a3f46; columns: 2; column-gap: 18px; }
.refs p { margin: 0 0 1px; break-inside: avoid; }

.tag { display: inline-block; font-size: 6.6pt; font-weight: 650; letter-spacing: .04em; text-transform: uppercase;
       border-radius: 3px; padding: 0 5px; white-space: nowrap; }
.tag.pass { background: #e6f4e6; color: #0f6b0f; } .tag.fixed { background: #fff1dc; color: #8a5300; }
.tag.finding { background: #eaf2fc; color: #1f63c4; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.card { border: 1px solid #e3e6ea; border-radius: 6px; padding: 5px 8px; font-size: 7.3pt; line-height: 1.38; break-inside: avoid; }
.card .top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }
.card .id { color: #5b616b; font: 7.2pt 'SF Mono', Menlo, monospace; }
.card .sc { font-weight: 650; font-size: 7.2pt; padding: 0 6px; border-radius: 3px; }
.card .sc.lo { background: #fdece6; color: #a8401a; } .card .sc.hi { background: #e6f4e6; color: #0f6b0f; }
.card .row { display: grid; grid-template-columns: 26px 1fr; gap: 4px; margin: 1px 0; }
.card .row b { font-size: 6.6pt; color: #5b616b; letter-spacing: .04em; padding-top: 1px; }
"""


def esc(s):
    return html.escape(str(s))


def f1(x):
    return f"{x:.1f}"


def svg_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


def style_axes(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9aa1aa")
    ax.tick_params(axis="both", labelsize=7, colors=MUTED, length=0, pad=3)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_length(an):
    """BLEU by source length: two thin lines, labelled at their right ends."""
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(3.55, 1.85), dpi=150)
    for name, color, label in [("opus100_test", IN_DOMAIN, "In-domain"), ("opus_books_sample", BOOKS, "Books")]:
        rows = an["by_length"][name]
        xs = list(range(len(rows)))
        ys = [r["bleu"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=1.2, solid_capstyle="round")
        ax.annotate(f"{label} {ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK, fontweight="bold")
    ax.set_xticks(range(len(rows)), [r["bucket"] for r in rows])
    ax.set_ylim(0, 40)
    ax.set_xlim(-0.25, len(rows) - 1 + 0.9)
    ax.set_xlabel("source length (words)", fontsize=7, color=MUTED, labelpad=3)
    ax.set_ylabel("BLEU", fontsize=7, color=MUTED, labelpad=3)
    style_axes(ax)
    return svg_uri(fig)


def chart_rare(an):
    """BLEU by share of rare source words: grouped bars from zero, values on the bars."""
    fig, ax = plt.subplots(figsize=(3.55, 1.85), dpi=150)
    buckets = [r["bucket"] for r in an["by_rare"]["opus100_test"]]
    w = 0.34
    for k, (name, color) in enumerate([("opus100_test", IN_DOMAIN), ("opus_books_sample", BOOKS)]):
        rows = an["by_rare"][name]
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(len(rows))]
        bars = ax.bar(xs, [r["bleu"] for r in rows], width=w, color=color, edgecolor="white", linewidth=0)
        for b, r in zip(bars, rows):
            ax.annotate(f"{r['bleu']:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(range(len(buckets)), [{"none": "none", "up to 10%": "up to 10%", "over 10%": "over 10%"}[b] for b in buckets])
    ax.set_ylim(0, 40)
    ax.set_xlabel("share of source words seen < 5 times in training", fontsize=7, color=MUTED, labelpad=3)
    ax.set_ylabel("BLEU", fontsize=7, color=MUTED, labelpad=3)
    style_axes(ax)
    return svg_uri(fig)


def _box(x, y, w, h, title, sub="", accent=False):
    fill, stroke = ("#eaf2fc", "#2a78d6") if accent else ("#ffffff", "#9aa1aa")
    t = (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='5' fill='{fill}' stroke='{stroke}' stroke-width='1'/>"
         f"<text x='{x + 9}' y='{y + h / 2 + (-1 if sub else 4)}' font-size='10.5' font-weight='600' fill='#1b1d21'>{title}</text>")
    if sub:
        t += f"<text x='{x + 9}' y='{y + h / 2 + 11}' font-size='9' fill='#5b616b'>{sub}</text>"
    return t


def arch_svg():
    """Encoder-decoder at a glance: where positions enter, how the two stacks connect, what is tied."""
    ex, dx, w, iw = 16, 380, 324, 300            # column x, width, inner-box width
    arrow = "marker-end='url(#ah)'"
    s = ["<svg viewBox='0 0 720 246' xmlns='http://www.w3.org/2000/svg' font-family='-apple-system, Helvetica Neue, Arial, sans-serif'>",
         "<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'>"
         "<path d='M0 0L10 5L0 10z' fill='#5b616b'/></marker></defs>"]
    # inputs and shared embeddings
    for x, label in [(ex, "French source tokens"), (dx, "English prefix (shifted right)")]:
        s.append(f"<text x='{x + w / 2}' y='16' font-size='10.5' text-anchor='middle' fill='#1b1d21' font-weight='600'>{label}</text>")
        s.append(f"<line x1='{x + w / 2}' y1='21' x2='{x + w / 2}' y2='31' stroke='#5b616b' {arrow}/>")
        s.append(_box(x, 33, w, 24, "Shared embedding &#215; &#8730;512", "", False))
        s.append(f"<line x1='{x + w / 2}' y1='57' x2='{x + w / 2}' y2='67' stroke='#5b616b' {arrow}/>")
    # encoder stack
    s.append(f"<rect x='{ex}' y='69' width='{w}' height='96' rx='7' fill='#f7f8fa' stroke='#1b1d21' stroke-width='1.2'/>")
    s.append(f"<text x='{ex + 10}' y='84' font-size='10.5' font-weight='700' fill='#1b1d21'>Encoder layer &#215; 6</text>")
    s.append(f"<text x='{ex + w - 10}' y='84' font-size='9' text-anchor='end' fill='#5b616b'>pre-LN, residual around each block</text>")
    s.append(_box(ex + 12, 92, iw, 32, "Self-attention, 8 heads", "RoPE rotates Q and K: attention sees relative distance", True))
    s.append(_box(ex + 12, 128, iw, 30, "Feed-forward 512 &#8594; 2048 &#8594; 512", "ReLU, dropout 0.1"))
    s.append(f"<line x1='{ex + w / 2}' y1='165' x2='{ex + w / 2}' y2='175' stroke='#5b616b' {arrow}/>")
    s.append(_box(ex, 177, w, 24, "Final LayerNorm &#8594; source memory"))
    # decoder stack
    s.append(f"<rect x='{dx}' y='69' width='{w}' height='126' rx='7' fill='#f7f8fa' stroke='#1b1d21' stroke-width='1.2'/>")
    s.append(f"<text x='{dx + 10}' y='84' font-size='10.5' font-weight='700' fill='#1b1d21'>Decoder layer &#215; 3</text>")
    s.append(f"<text x='{dx + w - 10}' y='84' font-size='9' text-anchor='end' fill='#5b616b'>shallow: runs once per output token</text>")
    s.append(_box(dx + 12, 92, iw, 30, "Causal self-attention", "RoPE on Q and K; sees only earlier tokens", True))
    s.append(_box(dx + 12, 126, iw, 30, "Cross-attention to source memory", "no positions: aligns by content"))
    s.append(_box(dx + 12, 160, iw, 28, "Feed-forward 512 &#8594; 2048 &#8594; 512"))
    s.append(f"<line x1='{dx + w / 2}' y1='195' x2='{dx + w / 2}' y2='205' stroke='#5b616b' {arrow}/>")
    s.append(_box(dx, 207, w, 32, "Final LN &#8594; output layer &#8594; softmax over 16k",
                  "output layer = embedding matrix (tied three ways)"))
    # encoder memory feeds every decoder layer's cross-attention
    s.append(f"<path d='M{ex + w} 189 H{ex + w + 22} V141 H{dx + 10}' fill='none' stroke='#2a78d6' stroke-width='1.4' {arrow}/>")
    s.append(f"<text x='{ex + w + 4}' y='206' font-size='9' fill='#1f63c4'>memory</text>")
    s.append("</svg>")
    return "".join(s)


def pipeline_svg():
    """End-to-end flow: data -> model -> selection on dev -> outputs and checks."""
    steps = [
        [("opus-100 en-fr train", "1,000,000 pairs"), ("Normalise + filter", "923,274 kept"),
         ("SentencePiece", "16k unigram, byte fallback"), ("Train 6/3 Transformer", "Colab T4, resumable"),
         ("Average checkpoints", "7k, 8k, 9k")],
        [("Tune decoding on dev", "beam x length penalty"), ("Decode test", "330 sentences"),
         ("Official score.py", "dev BLEU / chrF by slice"), ("Analysis", "CIs, length, rare words"),
         ("Publish", "GitHub, HF, W&B, report")],
    ]
    w, gap, h = 128, 20, 44
    arrow = "marker-end='url(#ph)'"
    s = ["<svg viewBox='0 0 720 150' xmlns='http://www.w3.org/2000/svg' font-family='-apple-system, Helvetica Neue, Arial, sans-serif'>",
         "<defs><marker id='ph' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'>"
         "<path d='M0 0L10 5L0 10z' fill='#5b616b'/></marker></defs>"]
    for r, row in enumerate(steps):
        y = 6 + r * 92
        for i, (t, sub) in enumerate(row):
            x = i * (w + gap)
            accent = (r, i) in [(0, 3), (1, 0)]
            fill, stroke = ("#eaf2fc", "#2a78d6") if accent else ("#ffffff", "#9aa1aa")
            s.append(f"<rect x='{x + 1}' y='{y}' width='{w - 2}' height='{h}' rx='6' fill='{fill}' stroke='{stroke}'/>")
            s.append(f"<text x='{x + w / 2}' y='{y + 19}' font-size='10' font-weight='600' text-anchor='middle' fill='#1b1d21'>{t}</text>")
            s.append(f"<text x='{x + w / 2}' y='{y + 33}' font-size='8.6' text-anchor='middle' fill='#5b616b'>{sub}</text>")
            if i < len(row) - 1:
                s.append(f"<line x1='{x + w}' y1='{y + h / 2}' x2='{x + w + gap - 1}' y2='{y + h / 2}' stroke='#5b616b' {arrow}/>")
    last = 4 * (w + gap) + w / 2
    s.append(f"<path d='M{last} 50 V74 H{w / 2} V96' fill='none' stroke='#5b616b' {arrow}/>")
    s.append("</svg>")
    return "".join(s)


def head(title, sub, eyebrow):
    chips = "".join(f"<a class=chip href='{u}'><b>{esc(k)}</b>{esc(v)}</a>" for k, v, u in LINKS)
    return (f"<div class=title><div class=eyebrow>{esc(eyebrow)}</div><h1>{title}</h1>"
            f"<p class=sub>{sub}</p><div class=chips>{chips}</div></div>")


def h2(n, text):
    return f"<h2><span class=n>{n}</span>{text}</h2>"


def build(run):
    rep = json.load(open(f"{run}/eval/dev_report.json"))
    sweep = json.load(open(f"{run}/eval/sweep.json"))
    an = json.load(open(f"{run}/analysis/analysis.json"))
    stats = json.load(open("work/data_stats.json"))
    sl = an["dev"]["by_slice"]
    gap = an["dev"]["gap_seen_minus_unseen"]
    aux, ds = an["aux_sets"], an["domain_stats"]
    o, b = ds["opus100_test"], ds["opus_books_sample"]
    rb = {name: {r["bucket"]: r for r in rows} for name, rows in an["by_rare"].items()}
    lb = {name: rows for name, rows in an["by_length"].items()}
    sw = {(r["beam"], r["alpha"]): r for r in sweep["results"] if r.get("overall") is not None}
    greedy, best = sw[(1, 0.0)], sw[(rep["decode"]["beam"], rep["decode"]["alpha"])]
    split = an["split_ablation"]
    aux_dec = "greedy" if an.get("aux_decode", {}).get("beam", 0) == 1 else "beam"
    none_gap = rb["opus100_test"]["none"]["bleu"] - rb["opus_books_sample"]["none"]["bleu"]

    def ci(v, key):
        lo, hi = v[key]
        return f"{lo:.1f} to {hi:.1f}".replace("-", "&minus;")

    slice_rows = "".join(
        f"<tr><td class=k>{k.replace('_', ' ')}</td><td class=n>{sl[k]['n']}</td><td class=n><b>{f1(sl[k]['bleu'])}</b></td>"
        f"<td class=ci>{ci(sl[k], 'bleu_ci')}</td><td class=n><b>{f1(sl[k]['chrf'])}</b></td><td class=ci>{ci(sl[k], 'chrf_ci')}</td></tr>"
        for k in ["seen", "long", "unseen_domain"])
    dec_rows = ""
    for key, label in [((1, 0.0), "Greedy"), ((4, 1.4), "Beam 4, &alpha; 1.4"), ((5, 1.4), "Beam 5, &alpha; 1.4"),
                       ((5, 1.8), "Beam 5, &alpha; 1.8 &mdash; chosen"), ((5, 2.2), "Beam 5, &alpha; 2.2")]:
        r = sw[key]
        cls = " class=hl" if key == (rep["decode"]["beam"], rep["decode"]["alpha"]) else ""
        dec_rows += f"<tr{cls}><td>{label}</td><td class=n>{r['overall']:.2f}</td><td class=n>{r['bleu']:.2f}</td><td class=n>{r['chrf']:.2f}</td></tr>"
    curve_rows = "".join(f"<tr><td class=n>{s:,}</td><td class=n>{v:.2f}</td><td class=n>{ov:.1f}</td></tr>" for s, v, ov in CURVE)

    body = f"""
{head("French&rarr;English Transformer from scratch: where it generalizes and where it breaks",
      "Encoder-decoder trained from scratch on filtered opus-100; evaluated in-domain, on long sentences, and on literature it never saw.",
      "Technical report")}

<div class=kpis>
 <div class="kpi accent"><div class=v>{rep['OVERALL']:.2f}</div><div class=l>Dev OVERALL score</div></div>
 <div class=kpi><div class=v>{f1(sl['seen']['bleu'])}<small> / {f1(sl['seen']['chrf'])}</small></div><div class=l>Seen &middot; BLEU / chrF</div></div>
 <div class=kpi><div class=v>{f1(sl['long']['bleu'])}<small> / {f1(sl['long']['chrf'])}</small></div><div class=l>Long &middot; BLEU / chrF</div></div>
 <div class="kpi warn"><div class=v>{f1(sl['unseen_domain']['bleu'])}<small> / {f1(sl['unseen_domain']['chrf'])}</small></div><div class=l>Books (unseen) &middot; BLEU / chrF</div></div>
 <div class=kpi><div class=v>39.7<small>M</small></div><div class=l>Params &middot; 3.6 h, one T4</div></div>
</div>

<div class=callout><div class=h>Key findings</div><ol>
<li><b>Length is not the failure mode; domain is.</b> Literature trails in-domain at every sentence length.</li>
<li><b>The dev gap needs more data to be trusted.</b> On 60 sentences its 95% CI includes zero; on 1,000 per domain it is
{f1(aux['opus100_test']['bleu'])} vs {f1(aux['opus_books_sample']['bleu'])} BLEU.</li>
<li><b>Register explains more than vocabulary.</b> Books sentences with no rare word still trail by {f1(none_gap)} BLEU; rare words cost more on top.</li>
</ol></div>

{h2(1, "Architecture decisions")}
<figure style='margin:0 0 4px'>{arch_svg()}
<div class=fs style='margin-top:2px'>39.7M parameters. Positions enter only through RoPE inside the two self-attentions; the encoder's output
(memory) feeds the cross-attention of every decoder layer; one 16k &times; 512 matrix serves as both embeddings and the output layer.</div></figure>
<table><tr><th style='width:23%'>Choice</th><th style='width:42%'>Why</th><th>Rejected alternative</th></tr>
<tr><td class=k>Encoder-decoder [1]</td><td>Reads the source bidirectionally; cross-attention aligns target to source words</td><td>Decoder-only: causal view of the source, sequences 2&times; longer</td></tr>
<tr><td class=k>6 enc / 3 dec, d 512 [2]</td><td><b>Measured</b> on the T4: 6/6 ran at 11.5k target tok/s, 6/3 at 13.6k, so ~20% more updates in budget</td><td>6/6 base: slightly better per step, 15% slower</td></tr>
<tr><td class=k>Pre-LayerNorm [3]</td><td>Stable gradients at a high peak LR (7e-4)</td><td>Post-LN: diverges easily early</td></tr>
<tr><td class=k>RoPE, self-attention only [4]</td><td>Relative positions; cross-attention aligns by content. Not chosen for extrapolation: <i>long</i> is within the trained range</td><td>Sinusoidal (ablation config), ALiBi [5]</td></tr>
<tr><td class=k>Tied embeddings [6]</td><td>One matrix for encoder input, decoder input and output layer; saves 16.4M params</td><td>Separate vocabularies</td></tr>
<tr><td class=k>SentencePiece 16k [7, 8]</td><td>Joint unigram vocab, byte fallback (no &lt;unk&gt;), source-side subword sampling &alpha;=0.1</td><td>BPE [9], 32k vocab, characters</td></tr>
<tr><td class=k>Beam 5, &alpha;={rep['decode']['alpha']} [10]</td><td>Tuned on dev; chrF (&beta;=2) and BLEU's brevity penalty both reward full-length output</td><td>Greedy: &minus;{best['overall'] - greedy['overall']:.1f} OVERALL</td></tr>
<tr><td class=k>Own PyTorch code</td><td>RoPE inside attention, pre-LN, tying; ~170 lines, unit-tested</td><td><code>nn.Transformer</code>: no RoPE hook</td></tr>
</table>

{h2(2, "Data and training")}
<div class=grid2><div>
<p><b>Data.</b> opus-100 en-fr <i>train</i> only, no subsampling. Unicode and quote normalisation (books sources use &rsquo;, their
references use '), then filtering kept <b>{stats['kept']:,} of {stats['input']:,}</b> pairs: {stats['duplicate']:,} duplicates,
{stats['bad_length_ratio']:,} bad length ratios, {stats['src_equals_tgt']:,} untranslated copies, {stats['empty_or_no_letters']:,} empty,
{stats['too_long_chars']:,} over-long, {stats['overlaps_dev_or_test']} dev/test overlaps. opus_books was never used for training,
the tokenizer or any tuning.</p>
<p><b>Training.</b> Label smoothing 0.1, AdamW (0.9, 0.98), 4k warmup + inverse-sqrt, ~24k tokens per update, fp16, seed 42.
Stopped at step 9,000 (~8.3 epochs, 3.6 of a planned 5.5 h) when the free-Colab GPU quota ran out, so the model is
<b>under-trained</b>. Final weights average checkpoints 7k&ndash;9k [11] (greedy 38.49 vs 38.14 for 9k alone).</p>
</div><div>
<table><tr><th class=n>Step</th><th class=n>Val loss</th><th class=n>Dev OVERALL (greedy)</th></tr>{curve_rows}</table>
<table><tr><th>Decoding (dev)</th><th class=n>OVERALL</th><th class=n>BLEU</th><th class=n>chrF</th></tr>{dec_rows}</table>
<p class=muted style='font-size:7.2pt'>Full 11-point grid in the appendix. Gains above &alpha; 1.4 are within noise.</p>
</div></div>

{h2(3, "Dev results by slice")}
<table><tr><th>Slice</th><th class=n>n</th><th class=n>BLEU</th><th class=n>95% CI</th><th class=n>chrF</th><th class=n>95% CI</th></tr>{slice_rows}
<tr class=total><td>All</td><td class=n>150</td><td class=n>{f1(rep['all']['bleu'])}</td><td></td><td class=n>{f1(rep['all']['chrf'])}</td><td></td></tr></table>
<p>OVERALL = 0.4&middot;BLEU + 0.4&middot;chrF + 0.2&middot;chrF<sub>unseen</sub> = <b>{rep['OVERALL']:.2f}</b>. Bootstrap CIs use 1,000 resamples.
The seen&minus;unseen gap is {f1(sl['seen']['bleu'] - sl['unseen_domain']['bleu'])} BLEU (CI {ci(gap, 'bleu_ci')}) and
{f1(sl['seen']['chrf'] - sl['unseen_domain']['chrf'])} chrF (CI {ci(gap, 'chrf_ci')}): <b>not significant on 60 sentences</b>, so section 4
re-measures it on 1,000. <i>Long</i> beats <i>seen</i> because long opus-100 sentences are formulaic official text, while short ones are
colloquial subtitles with loose references (about 4 of 150 dev pairs are misaligned).</p>

{h2(4, "Generalization: evidence and hypothesis")}
<table><tr><th>Sample (n=1,000 each, {aux_dec})</th><th class=n>BLEU</th><th class=n>chrF</th><th class=n>Unseen words</th>
<th class=n>Rare words</th><th class=n>Ref. loss</th><th class=n>Out/ref length</th></tr>
<tr><td class=k><span style='color:{IN_DOMAIN}'>&#9632;</span> opus-100 test (in-domain)</td><td class=n>{f1(aux['opus100_test']['bleu'])}</td><td class=n>{f1(aux['opus100_test']['chrf'])}</td>
<td class=n>{o['oov_token_rate']:.1%}</td><td class=n>{o['rare_token_rate']:.1%}</td><td class=n>{o['ref_loss_per_token']:.2f}</td><td class=n>{o['hyp_ref_len_ratio']:.2f}</td></tr>
<tr><td class=k><span style='color:{BOOKS}'>&#9632;</span> opus_books sample (unseen)</td><td class=n>{f1(aux['opus_books_sample']['bleu'])}</td><td class=n>{f1(aux['opus_books_sample']['chrf'])}</td>
<td class=n>{b['oov_token_rate']:.1%}</td><td class=n>{b['rare_token_rate']:.1%}</td><td class=n>{b['ref_loss_per_token']:.2f}</td><td class=n>{b['hyp_ref_len_ratio']:.2f}</td></tr></table>

<div class=key><span class=a>In-domain (opus-100 test)</span><span class=b>Books (opus_books)</span></div>
<div class=grid2>
<figure><div class=ft>Books trail at every length</div><div class=fs>BLEU by source length; in-domain stays flat even past 45 words</div>
<img src='{chart_length(an)}'></figure>
<figure><div class=ft>Rare words hurt only in books</div><div class=fs>BLEU by share of rare source words; rare-free books already lag</div>
<img src='{chart_rare(an)}'></figure>
</div>

<div class=finding><div class=i>1</div><div><b>Register is the largest share.</b> Books sentences with no rare word score
{f1(rb['opus_books_sample']['none']['bleu'])} vs {f1(rb['opus100_test']['none']['bleu'])} BLEU in-domain, and the model's loss on the correct
reference is {b['ref_loss_per_token']:.2f} vs {o['ref_loss_per_token']:.2f} nats/token (perplexity {b['ref_ppl']:.1f} vs {o['ref_ppl']:.1f}):
trained on subtitles and official text, it finds 19th-century narrative English improbable.</div></div>
<div class=finding><div class=i>2</div><div><b>Rare vocabulary adds to it.</b> Books have {b['oov_token_rate'] / o['oov_token_rate']:.1f}&times; the
never-seen-word rate and {b['rare_token_rate'] / o['rare_token_rate']:.1f}&times; the rare-word rate; the
{rb['opus_books_sample']['over 10%']['n']} books sentences with over 10% rare words fall to {f1(rb['opus_books_sample']['over 10%']['bleu'])} BLEU.</div></div>
<div class=finding><div class=i>3</div><div><b>Freer references and shorter outputs.</b> Output/reference length is {b['hyp_ref_len_ratio']:.2f} on books
vs {o['hyp_ref_len_ratio']:.2f}, so BLEU's brevity penalty bites harder. Typography is ruled out by normalisation.</div></div>

{h2(5, "Where it fails")}
<table><tr><th style='width:24%'>Failure</th><th>Examples and frequency</th></tr>
<tr><td class=k>Names, rare words</td><td>Subword pieces recombine into a frequent word: <i>Le grand Meaulnes</i> &rarr; &ldquo;The Great Meals&rdquo;,
<i>roulotte</i> &rarr; &ldquo;roller&rdquo;, <i>grognait</i> &rarr; &ldquo;grew up&rdquo;</td></tr>
<tr><td class=k>Literary word senses</td><td>The common sense wins: <i>cils</i> (eyelashes) &rarr; &ldquo;cloaks&rdquo;, <i>fers</i> (forceps) &rarr; &ldquo;irons&rdquo;.
Past tenses mostly right (<i>r&eacute;pondit</i> &rarr; &ldquo;answered&rdquo;), with slips (<i>jeta</i> &rarr; &ldquo;is throwing&rdquo;)</td></tr>
<tr><td class=k>Repetition loops</td><td>Rare: 0.5% of in-domain and 0.4% of books outputs repeat a 3-gram three or more times</td></tr>
<tr><td class=k>Very long inputs</td><td>Inputs over 128 subword tokens are split at sentence ends: on {split['n']} such sentences, chrF
{f1(split['split']['chrf'])} split vs {f1(split['no_split']['chrf'])} unsplit</td></tr>
<tr><td class=k>Reference noise</td><td>Misaligned opus-100 pairs and books references with stray quotes or chapter headings cap the scores</td></tr>
</table>

{h2(6, "Challenges and how they were resolved")}
<table><tr><th style='width:30%'>Problem</th><th>Resolution</th></tr>
<tr><td class=k>Colab disconnects; GPU quota ended training at 3.6 h</td><td>Atomic full-state checkpoints, deterministic batch order, wall-clock budget across sessions, same W&amp;B run.
A planned kill-and-resume test caught a bug (RNG state moved to GPU) before the real run</td></tr>
<tr><td class=k>T4 slower than planned</td><td>Switched 6/6 &rarr; 6/3 on a throughput threshold fixed before measuring</td></tr>
<tr><td class=k>Noisy data; curly vs straight quotes</td><td>Filtering (7.7% removed); one normaliser on inputs and outputs (one &rsquo; costs ~20 chrF on a sentence)</td></tr>
<tr><td class=k>Decoding stalls on one long input</td><td>Length-sorted batches capped at 2,048 source tokens; outputs unchanged</td></tr>
<tr><td class=k>Tokenizer differs across machines</td><td>12 of 16k pieces changed with the thread count; pinned to 2 threads, and the trained tokenizer ships with the model</td></tr>
<tr><td class=k>Beam-search and splitter edge cases</td><td>Unit tests: greedy = beam(1), overfit-and-recover, abbreviation-aware splitting</td></tr>
</table>

{h2(7, "How the results were validated")}
<table><tr><th style='width:26%'>Check</th><th>Evidence</th></tr>
<tr><td class=k>Model correctness</td><td>5 unit tests: causal mask, padding invariance (RoPE and sinusoidal), a tiny model memorises 8 pairs and
greedy, beam(1) and beam(4) all recover them, normalisation, sentence splitting</td></tr>
<tr><td class=k>Metric fidelity</td><td>Scoring imports the official <code>score.py</code>; identical OVERALL (39.94) to running it directly</td></tr>
<tr><td class=k>No leakage</td><td>Dev/test sources removed from training; opus_books used only after every choice was fixed; all tuning on dev</td></tr>
<tr><td class=k>Reproducibility</td><td>Identical filter counts on two machines; seeded batch order; one command reproduces the run</td></tr>
<tr><td class=k>Uncertainty</td><td>Bootstrap CIs on every dev slice; the gap re-measured on 1,000 sentences per domain</td></tr>
<tr><td class=k>Claims vs data</td><td>Each failure claim counted on 2,000 outputs before writing; two over-strong draft claims were corrected</td></tr>
<tr><td class=k>Deliverables</td><td>Submission matches all 330 ids; Hugging Face weights match by SHA-256 and load with 3 dependencies</td></tr>
</table>
<p class=muted style='font-size:7.2pt'>All 19 checks, with results, are in the appendix.</p>

{h2(8, "Next steps with more time or compute")}
<ul><li><b>Train to convergence</b> (dev still rising at 9k steps) with a cosine tail, and several seeds for variance.</li>
<li><b>Close the domain gap:</b> back-translate monolingual English literature (not opus_books) [12], domain tags, a copy mechanism for names.</li>
<li><b>Run the prepared ablations</b> (RoPE vs sinusoidal, 16k vs 32k vocab) and stronger data filtering [13].</li>
<li><b>Decoding:</b> KV cache plus MBR with chrF as the utility; report sacreBLEU alongside the challenge scorer.</li></ul>

<h2 style='margin-top:10px'>References</h2>
<div class=refs>
<p>[1] Vaswani et al. 2017, Attention Is All You Need</p><p>[2] Kasai et al. 2021, Deep Encoder, Shallow Decoder</p>
<p>[3] Xiong et al. 2020, On Layer Normalization in the Transformer</p><p>[4] Su et al. 2021, RoFormer: Rotary Position Embedding</p>
<p>[5] Press et al. 2022, Train Short, Test Long (ALiBi)</p><p>[6] Press &amp; Wolf 2017, Using the Output Embedding</p>
<p>[7] Kudo &amp; Richardson 2018, SentencePiece</p><p>[8] Kudo 2018, Subword Regularization</p>
<p>[9] Sennrich et al. 2016, NMT of Rare Words with Subword Units</p><p>[10] Wu et al. 2016, Google's NMT System</p>
<p>[11] Popel &amp; Bojar 2018, Training Tips for the Transformer</p><p>[12] Sennrich et al. 2016, NMT with Monolingual Data</p>
<p>[13] Junczys-Dowmunt 2018, Dual Conditional Cross-Entropy Filtering</p></div>
"""
    return page("FR-EN Transformer from scratch: report", body)


def page(title, body):
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


TAGS = {"Tokenizer reproducibility": "fixed", "Resume correctness": "fixed", "Failure-mode claims": "fixed",
        "Statistical uncertainty": "finding", "Throughput-based model choice": "finding", "Checkpoint selection": "finding",
        "Decoding selection": "finding", "Long-input policy": "finding"}


def arrows(text):
    return esc(text).replace(" -&gt; ", " &rarr; ")


def parse_examples(md):
    """examples.md -> [(slice, [(score, id, src, ref, hyp), ...]), ...]"""
    out = []
    for block in md.split("## ")[1:]:
        title, *items = block.strip().split("\n- ")
        exs = []
        for it in items:
            lines = [x.strip().lstrip("- ") for x in it.split("\n")]
            score, _, sid = lines[0].replace("**", "").replace("`", "").strip().partition(" ")
            fields = {x.split(":", 1)[0]: x.split(":", 1)[1].strip() for x in lines[1:] if ":" in x}
            exs.append((int(float(score)), sid.strip(), fields.get("SRC", ""), fields.get("REF", ""), fields.get("HYP", "")))
        out.append((title.split(":")[0].strip(), exs))
    return out


def build_appendix(run, validations):
    an = json.load(open(f"{run}/analysis/analysis.json"))
    sweep = json.load(open(f"{run}/eval/sweep.json"))
    rep = json.load(open(f"{run}/eval/dev_report.json"))
    chosen = (rep["decode"]["beam"], rep["decode"]["alpha"])

    val_rows = ""
    for check, what, result in validations:
        t = TAGS.get(check, "pass")
        val_rows += (f"<tr><td class=k>{esc(check)}<br><span class='tag {t}'>{t}</span></td>"
                     f"<td>{arrows(what)}</td><td>{arrows(result)}</td></tr>")

    sweep_rows = ""
    for r in sweep["results"]:
        if r.get("overall") is None:
            continue
        cls = " class=hl" if (r["beam"], r["alpha"]) == chosen else ""
        name = "Greedy" if r["beam"] == 1 else f"Beam {r['beam']}"
        sweep_rows += (f"<tr{cls}><td>{name}</td><td class=n>{r['alpha'] if r['beam'] > 1 else '&ndash;'}</td>"
                       f"<td class=n>{r['overall']:.2f}</td><td class=n>{r['bleu']:.2f}</td><td class=n>{r['chrf']:.2f}</td></tr>")

    def bucket_table(key, name, label):
        rows = "".join(f"<tr><td>{esc(r['bucket'])}</td><td class=n>{r['n']}</td><td class=n>{f1(r['bleu'])}</td><td class=n>{f1(r['chrf'])}</td></tr>"
                       for r in an[key][name])
        return f"<table><tr><th>{label}</th><th class=n>n</th><th class=n>BLEU</th><th class=n>chrF</th></tr>{rows}</table>"

    cards_html = ""
    for slice_name, exs in parse_examples(open(f"{run}/analysis/examples.md").read()):
        cards = ""
        for score, sid, src, ref, hyp in exs:
            cls = "lo" if score < 50 else "hi"
            cards += (f"<div class=card><div class=top><span class=id>{esc(sid)}</span><span class='sc {cls}'>chrF {score}</span></div>"
                      f"<div class=row><b>SRC</b><span>{esc(src)}</span></div><div class=row><b>REF</b><span>{esc(ref)}</span></div>"
                      f"<div class=row><b>OUT</b><span>{esc(hyp)}</span></div></div>")
        cards_html += f"<h3>{esc(slice_name.replace('_', ' ').capitalize())}: four lowest, two highest</h3><div class=cards>{cards}</div>"

    body = f"""
{head("Appendix: validations and work beyond the brief",
      "Companion to the 3-page report. Every number comes from files in the repo or the Hugging Face model repo (<code>eval/</code>, <code>analysis/</code>), regenerated by <code>python report/build_report.py</code>.",
      "Appendix")}

<figure style='margin:2px 0 4px'><div class=ft>End-to-end pipeline</div>
<div class=fs>Everything right of training uses dev only for choices; the analysis runs after every choice is fixed. Blue: the two steps that set the model and its decoding.</div>
{pipeline_svg()}</figure>

{h2("A", "Work beyond the brief")}
<table><tr><th style='width:30%'>Item</th><th>What it adds</th></tr>
<tr><td class=k>Eval-only analysis, 2 &times; 1,000 sentences</td><td>The 60-sentence dev slices cannot show the gap is real; the larger samples, length and rare-word buckets and the reference loss separate length, vocabulary and register</td></tr>
<tr><td class=k>Bootstrap confidence intervals</td><td>Per dev slice and for the seen&minus;unseen gap</td></tr>
<tr><td class=k>Checkpoint and decoding studies</td><td>3 checkpoint sets; an 11-point beam &times; length-penalty grid, extended when the optimum sat at the grid edge</td></tr>
<tr><td class=k>Resumable training for free Colab</td><td>Atomic full-state checkpoints, deterministic batch order, wall-clock budget across sessions; survived two interruptions</td></tr>
<tr><td class=k>Self-contained Hugging Face package</td><td>Weights (safetensors), tokenizer, inference code, model card with real metrics; loads with 3 dependencies</td></tr>
<tr><td class=k>W&amp;B report</td><td>Loss, LR, throughput, dev BLEU/chrF per slice, sample translations and config (<code>scripts/wandb_report.py</code>)</td></tr>
<tr><td class=k>Colab notebook and setup guide</td><td>Secrets, Drive checkpoints, throughput check, analysis and Hub-push cells (<code>docs/COLAB_SETUP.md</code>)</td></tr>
<tr><td class=k>Reproducible reporting</td><td>This appendix and the report are generated from result files by one script</td></tr>
<tr><td class=k>Prepared ablations</td><td>RoPE vs sinusoidal and 6/6 configs ready to run (<code>configs/</code>)</td></tr>
</table>

{h2("B", "Validations performed, with evidence")}
<p class=muted style='font-size:7.4pt'><span class='tag pass'>pass</span> check passed &nbsp; <span class='tag fixed'>fixed</span> found a problem that was fixed
&nbsp; <span class='tag finding'>finding</span> measurement that drove a decision</p>
<table class=long><tr><th style='width:21%'>Check</th><th style='width:47%'>What was done</th><th>Result</th></tr>{val_rows}</table>

{h2("C", "Decoding grid and score breakdowns")}
<div class=grid2><div>
<table><tr><th>Decoding (dev, avg 7k&ndash;9k)</th><th class=n>&alpha;</th><th class=n>OVERALL</th><th class=n>BLEU</th><th class=n>chrF</th></tr>{sweep_rows}</table>
<p class=muted style='font-size:7.2pt'>Beam 8 was not run: memory pressure on the laptop, and beam 4 &rarr; 5 gained under 0.4.</p>
</div><div>
{bucket_table("by_length", "opus100_test", "In-domain, by length (words)")}
{bucket_table("by_length", "opus_books_sample", "Books, by length (words)")}
</div></div>
<div class=grid2><div>{bucket_table("by_rare", "opus100_test", "In-domain, by share of rare words")}</div>
<div>{bucket_table("by_rare", "opus_books_sample", "Books, by share of rare words")}</div></div>

{h2("D", "Lowest- and highest-scoring dev sentences per slice")}
{cards_html}
"""
    return page("FR-EN Transformer from scratch: appendix", body)


def pdf(chrome, here, name):
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={here}/{name}.pdf", f"file://{here}/{name}.html"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {here}/{name}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="checkpoints/base_6x3")
    ap.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(f"{here}/report.html", "w") as fh:
        fh.write(build(a.run))
    pdf(a.chrome, here, "report")
    from validations import VALIDATIONS
    with open(f"{here}/appendix.html", "w") as fh:
        fh.write(build_appendix(a.run, VALIDATIONS))
    pdf(a.chrome, here, "appendix")


if __name__ == "__main__":
    main()
