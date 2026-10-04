"""Build report/report.pdf from the result files, so every number in the report is traceable.

    python report/build_report.py --run checkpoints/base_6x3 [--chrome /path/to/chrome]

Reads: <run>/eval/dev_report.json, <run>/eval/sweep.json, <run>/analysis/analysis.json (+ PNGs),
       work/data_stats.json. Writes report/report.html and report/report.pdf (headless Chrome).
"""
import argparse
import base64
import html
import json
import os
import subprocess

LINKS = {
    "GitHub": "https://github.com/bad2212/transfromer-model",
    "Hugging Face": "https://huggingface.co/Badalt/fr-en-transformer-scratch",
    "W&B run": "https://wandb.ai/badalthakur2212-iisc/fr-en-transformer/runs/5ks4crpz",
    "W&B report": "https://wandb.ai/badalthakur2212-iisc/fr-en-transformer/reports/FR%E2%86%92EN-Transformer-from-scratch:-training-and-generalization--VmlldzoxODA1MDg0Mw==",
}
# Greedy dev evaluations logged during training (W&B run 5ks4crpz), single checkpoint.
CURVE = [(2000, 4.11, 14.45), (4000, 2.65, 30.79), (6000, 2.28, 35.39), (8000, 2.14, 37.58)]

CSS = """
@page { size: A4; margin: 13mm 14mm 12mm 14mm; }
body { font: 8.6pt/1.33 -apple-system, 'Helvetica Neue', Arial, sans-serif; color: #1a1a1a; margin: 0; }
h1 { font-size: 14pt; margin: 0 0 2px; } h2 { font-size: 10pt; margin: 9px 0 3px; border-bottom: 1px solid #ccc; padding-bottom: 1px; }
p { margin: 0 0 4px; } ul { margin: 0 0 4px; padding-left: 15px; } li { margin: 0 0 1px; }
table { border-collapse: collapse; width: 100%; margin: 3px 0 5px; font-size: 7.9pt; }
th, td { border-bottom: 1px solid #e2e2e2; padding: 1.5px 4px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; } td.n, th.n { text-align: right; white-space: nowrap; }
.links { font-size: 7.8pt; color: #333; margin-bottom: 5px; } a { color: #1f5fbf; text-decoration: none; }
.two { display: flex; gap: 10px; } .two > div { flex: 1; min-width: 0; }
img { width: 100%; } .cap { font-size: 7.3pt; color: #555; margin: 0 0 4px; }
.refs { font-size: 7.3pt; columns: 2; column-gap: 14px; } .refs p { margin: 0 0 1px; }
b.k { font-weight: 600; }
"""


def esc(s):
    return html.escape(str(s))


def f1(x):
    return f"{x:.1f}"


def ci(v, key):
    lo, hi = v[key]
    return f"[{lo:.1f}, {hi:.1f}]"


def img(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def build(run):
    rep = json.load(open(f"{run}/eval/dev_report.json"))
    sweep = json.load(open(f"{run}/eval/sweep.json"))
    an = json.load(open(f"{run}/analysis/analysis.json"))
    stats = json.load(open("work/data_stats.json"))
    d = an["dev"]
    sl = d["by_slice"]
    gap = d["gap_seen_minus_unseen"]
    aux, ds = an["aux_sets"], an["domain_stats"]
    o, b = ds["opus100_test"], ds["opus_books_sample"]
    sw = {(r["beam"], r["alpha"]): r for r in sweep["results"] if r.get("overall") is not None}
    greedy, best = sw[(1, 0.0)], sw[(rep["decode"]["beam"], rep["decode"]["alpha"])]
    split = an.get("split_ablation")
    rb = {name: {r["bucket"]: r for r in rows} for name, rows in an["by_rare"].items()}
    rare_none_gap = rb["opus100_test"]["none"]["bleu"] - rb["opus_books_sample"]["none"]["bleu"]
    aux_dec = "greedy" if an.get("aux_decode", {}).get("beam", 0) == 1 else "beam"

    slice_rows = "".join(
        f"<tr><td>{k}</td><td class=n>{sl[k]['n']}</td><td class=n>{f1(sl[k]['bleu'])}</td><td class=n>{ci(sl[k], 'bleu_ci')}</td>"
        f"<td class=n>{f1(sl[k]['chrf'])}</td><td class=n>{ci(sl[k], 'chrf_ci')}</td></tr>"
        for k in ["seen", "long", "unseen_domain"])
    sweep_rows = "".join(
        f"<tr><td>{r['beam']}</td><td class=n>{r['alpha']}</td><td class=n>{r['overall']:.2f}</td><td class=n>{r['bleu']:.2f}</td><td class=n>{r['chrf']:.2f}</td></tr>"
        for r in sweep["results"] if r.get("overall") is not None)
    curve_rows = "".join(f"<tr><td class=n>{s:,}</td><td class=n>{v:.2f}</td><td class=n>{ov:.1f}</td></tr>" for s, v, ov in CURVE)
    split_txt = ""
    if split:
        split_txt = (f"On the {split['n']} eval-only sentences over 128 subword tokens, splitting gives "
                     f"chrF {f1(split['split']['chrf'])} vs {f1(split['no_split']['chrf'])} without it "
                     f"(BLEU {f1(split['split']['bleu'])} vs {f1(split['no_split']['bleu'])}).")
    label = {"GitHub": "bad2212/transfromer-model", "Hugging Face": "Badalt/fr-en-transformer-scratch",
             "W&B run": "fr-en-transformer / run 5ks4crpz", "W&B report": "training and generalization report"}
    links = " &nbsp;·&nbsp; ".join(f"<b class=k>{esc(k)}:</b> <a href='{v}'>{esc(label[k])}</a>" for k, v in LINKS.items())

    return f"""<!doctype html><html><head><meta charset='utf-8'><title>FR-EN Transformer from scratch</title><style>{CSS}</style></head><body>
<h1>French→English Transformer trained from scratch: where it generalizes and where it breaks</h1>
<div class=links>{links}</div>

<h2>1. Summary</h2>
<p>A 39.7M-parameter encoder-decoder Transformer (6 encoder / 3 decoder layers, pre-LN, RoPE, tied embeddings, 16k joint
SentencePiece) trained from scratch on {stats['kept']:,} filtered opus-100 pairs for 3.6 h on one Colab T4. Dev OVERALL
<b class=k>{rep['OVERALL']:.2f}</b>: BLEU {f1(sl['seen']['bleu'])} on <i>seen</i>, {f1(sl['long']['bleu'])} on <i>long</i>,
{f1(sl['unseen_domain']['bleu'])} on <i>unseen_domain</i> (literature). <b class=k>Length is not the failure mode at these lengths;
domain is.</b> The 60-sentence dev slices are too small to prove the gap (its 95% CI includes zero), but 1,000-sentence eval-only samples
show it clearly: {f1(aux['opus100_test']['bleu'])} vs {f1(aux['opus_books_sample']['bleu'])} BLEU. Rare words explain part of it; most of it
is register: even books sentences with no rare word trail in-domain ones by ~{f1(rare_none_gap)} BLEU, and the model's loss on literary references is
{b['ref_loss_per_token'] - o['ref_loss_per_token']:.2f} nats/token higher.</p>

<h2>2. Architecture decisions and the alternatives rejected</h2>
<table><tr><th>Choice</th><th>Why</th><th>Rejected alternative (trade-off)</th></tr>
<tr><td>Encoder-decoder [1]</td><td>Conditional generation: bidirectional source encoding, cross-attention aligns target to source words</td><td>Decoder-only: causal view of the source, sequences 2x longer, capacity spent modelling French</td></tr>
<tr><td>6 enc / 3 dec, d=512, 8 heads, FFN 2048 [1, 2]</td><td><b class=k>Measured on the T4:</b> 6/6 (52.3M) ran at 11.5k target tok/s, 6/3 at 13.6k: ~20% more updates in the budget; a shallow decoder also decodes faster [2]</td><td>6/6 base: slightly better per step, 15% slower; Transformer-big: ~4x compute, overfits 1M pairs</td></tr>
<tr><td>Pre-LayerNorm + final LN [3]</td><td>Stable gradients with a high peak LR (7e-4), no fragile warmup tuning</td><td>Post-LN: marginally better when tuned, diverges easily early</td></tr>
<tr><td>RoPE in self-attention; none in cross-attention [4]</td><td>Relative positions; cross-attention aligns by content. <i>Not</i> chosen for extrapolation: 8.5% of training pairs exceed 40 words, so <i>long</i> is in range</td><td>Sinusoidal (absolute; kept as ablation config), ALiBi [5] (extrapolation is not what <i>long</i> tests), learned (hard length cap)</td></tr>
<tr><td>Three-way tied embeddings [6]</td><td>Joint vocab: one matrix for encoder input, decoder input and output layer; saves 16.4M params, regularizes</td><td>Separate vocabularies: no tying, names/numbers harder to copy</td></tr>
<tr><td>SentencePiece unigram, 16k joint, byte fallback [7]; source-side subword sampling α=0.1 [8]</td><td>Robust segmentation of rare/unseen words; no &lt;unk&gt; ever; target stays deterministic</td><td>BPE [9] (needs BPE-dropout for the same effect), 32k (fewer examples per rare piece), characters (4-5x longer sequences)</td></tr>
<tr><td>Beam 5, GNMT length penalty α={rep['decode']['alpha']} [10]; split inputs over 128 tokens at sentence ends</td><td>Tuned on dev. chrF (β=2) and BLEU's brevity penalty both punish short output, so a long-output bias pays</td><td>Greedy (−{best['overall'] - greedy['overall']:.1f} OVERALL); KV cache omitted on purpose (500 sentences; most bug-prone code)</td></tr>
<tr><td>Own PyTorch implementation</td><td>RoPE inside attention, pre-LN, tying; ~170 lines, unit-tested (causality, padding, overfit-and-decode)</td><td><code>torch.nn.Transformer</code>: post-LN, no RoPE hook</td></tr>
</table>

<h2>3. Data and training</h2>
<p><b class=k>Data.</b> opus-100 en-fr train only. Normalised (NFKC; typographic quotes to ASCII, which matters because books sources use
’ while their references use '), then filtered: {stats['duplicate']:,} duplicates, {stats['bad_length_ratio']:,} bad length ratios,
{stats['src_equals_tgt']:,} untranslated copies, {stats['empty_or_no_letters']:,} empty, {stats['too_long_chars']:,} over 1,000 chars, and
{stats['overlaps_dev_or_test']} pairs whose source appears in dev/test: <b class=k>{stats['kept']:,} of {stats['input']:,} kept</b> (no subsampling).
opus_books was never used for training, the tokenizer, checkpoint choice or decoding tuning; a 1,000-sentence sample (dev/test sentences removed)
is used only post hoc in §5.</p>
<p><b class=k>Training.</b> Label smoothing 0.1 [1], AdamW (0.9, 0.98), 4k warmup + inverse-sqrt, ~24k tokens/update (8k × 3 accumulation),
length-bucketed batches, fp16, clip 1.0, seed 42 everywhere. Stopped at step 9,000 (≈8.3 epochs, 3.6 h of a planned 5.5 h) when the free-Colab GPU quota ran out;
dev was still improving (table below), so this model is under-trained. Final weights = average of checkpoints 7k-9k [11]: greedy OVERALL 38.49 vs 38.14 for step 9k
alone and 37.66 for averaging 5k-9k (early weights hurt while the model still improves fast).</p>
<div class=two><div><table><tr><th class=n>Step</th><th class=n>Val loss</th><th class=n>Dev OVERALL (greedy)</th></tr>{curve_rows}</table></div>
<div><table><tr><th>Beam</th><th class=n>α</th><th class=n>OVERALL</th><th class=n>BLEU</th><th class=n>chrF</th></tr>{sweep_rows}</table></div></div>

<h2>4. Dev results by slice (official scorer, final decoding)</h2>
<table><tr><th>Slice</th><th class=n>n</th><th class=n>BLEU</th><th class=n>95% CI</th><th class=n>chrF</th><th class=n>95% CI</th></tr>{slice_rows}
<tr><td><b class=k>all</b></td><td class=n>150</td><td class=n><b class=k>{f1(rep['all']['bleu'])}</b></td><td></td><td class=n><b class=k>{f1(rep['all']['chrf'])}</b></td><td></td></tr></table>
<p><b class=k>OVERALL = {rep['OVERALL']:.2f}</b>. CIs: 1,000 bootstrap resamples per slice. The seen − unseen_domain gap is
{f1(sl['seen']['bleu'] - sl['unseen_domain']['bleu'])} BLEU, 95% CI {ci(gap, 'bleu_ci')}, and
{f1(sl['seen']['chrf'] - sl['unseen_domain']['chrf'])} chrF, CI {ci(gap, 'chrf_ci')}: <b class=k>with 60 sentences per slice the dev gap
alone is not significant</b>, which is why §5 measures it on 1,000 sentences per domain.
<i>long</i> scores <i>above</i> <i>seen</i>: long opus-100 sentences are formulaic official text that is well covered in training and within the trained length,
while short <i>seen</i> sentences are colloquial subtitles with loose references (≈4 of 150 dev pairs are misaligned, e.g. dev_00012 → "Outside, Roger.").</p>

<h2>5. Generalization: evidence and hypothesis</h2>
<table><tr><th>Eval-only set (n=1,000 each, post hoc, {aux_dec})</th><th class=n>BLEU</th><th class=n>chrF</th><th class=n>src words</th><th class=n>OOV words</th><th class=n>rare (&lt;5) words</th><th class=n>sent. w/ rare</th><th class=n>ref loss/token</th><th class=n>hyp/ref length</th></tr>
<tr><td>opus-100 test (in-domain)</td><td class=n>{f1(aux['opus100_test']['bleu'])}</td><td class=n>{f1(aux['opus100_test']['chrf'])}</td><td class=n>{o['mean_src_words']:.1f}</td><td class=n>{o['oov_token_rate']:.1%}</td><td class=n>{o['rare_token_rate']:.1%}</td><td class=n>{o['sent_with_rare_word']:.0%}</td><td class=n>{o['ref_loss_per_token']:.2f}</td><td class=n>{o['hyp_ref_len_ratio']:.2f}</td></tr>
<tr><td>opus_books sample (unseen domain)</td><td class=n>{f1(aux['opus_books_sample']['bleu'])}</td><td class=n>{f1(aux['opus_books_sample']['chrf'])}</td><td class=n>{b['mean_src_words']:.1f}</td><td class=n>{b['oov_token_rate']:.1%}</td><td class=n>{b['rare_token_rate']:.1%}</td><td class=n>{b['sent_with_rare_word']:.0%}</td><td class=n>{b['ref_loss_per_token']:.2f}</td><td class=n>{b['hyp_ref_len_ratio']:.2f}</td></tr></table>
<div class=two><div><img src='{img(f"{run}/analysis/length.png")}'><p class=cap>chrF by source length. The books curve sits below opus-100 in every bucket, so length does not explain the gap.</p></div>
<div><img src='{img(f"{run}/analysis/rare.png")}'><p class=cap>chrF by share of source words seen fewer than 5 times in training. In-domain, rare words barely matter; in books, sentences with over 10% rare words fall sharply, yet rare-free books sentences already sit far below.</p></div></div>
<p><b class=k>Hypothesis.</b> The gap is register and lexical shift, not length. (1) <b class=k>Register (largest share)</b>: books sentences with <i>no</i> rare word
still score {f1(rb['opus_books_sample']['none']['bleu'])} BLEU vs {f1(rb['opus100_test']['none']['bleu'])} in-domain, and the model's loss on the correct reference is
{b['ref_loss_per_token']:.2f} vs {o['ref_loss_per_token']:.2f} nats/token (perplexity {b['ref_ppl']:.1f} vs {o['ref_ppl']:.1f}): after training on subtitles and official text,
19th-century narrative English is improbable to it. (2) <b class=k>Vocabulary</b>: books sources have {b['oov_token_rate'] / o['oov_token_rate']:.1f}x the never-seen-word rate
and {b['rare_token_rate'] / o['rare_token_rate']:.1f}x the rare-word rate; the {rb['opus_books_sample']['over 10%']['n']} books sentences with over 10% rare words drop to
{f1(rb['opus_books_sample']['over 10%']['bleu'])} BLEU. Names and literary words become frequent words ("Le grand Meaulnes" → "The Great Meals", <i>cils</i> (eyelashes) → "cloaks").
(3) <b class=k>Freer references and shorter outputs</b>: hypothesis/reference length is {b['hyp_ref_len_ratio']:.2f} on books vs {o['hyp_ref_len_ratio']:.2f}, so the brevity penalty bites more.
Typography is ruled out by normalisation. {split_txt}</p>

<h2>6. Where it fails</h2>
<ul><li><b class=k>Rare words and names</b>: subword pieces recombine into a frequent English word ("roulotte" → "roller", "grognait" → "grew up").</li>
<li><b class=k>Literary word senses</b>: the common sense wins over the literary one (<i>fers</i>, forceps → "irons"). Past tenses are mostly right
(<i>répondit</i> → "answered", <i>comprit</i> → "understood"), with occasional slips (<i>jeta</i> → "is throwing").</li>
<li><b class=k>Repetition loops are rare</b>: 0.5% of in-domain and 0.4% of books outputs repeat a 3-gram three times or more, mostly on long technical or literary inputs.</li>
<li><b class=k>Noisy references</b> cap scores: misaligned opus-100 pairs, books references with stray quotes and chapter headings.</li></ul>

<h2>7. Challenges and how they were resolved</h2>
<table><tr><th>Problem</th><th>Resolution</th></tr>
<tr><td>Free Colab disconnects, then the GPU quota ran out at 3.6 h</td><td>Atomic full-state checkpoints, deterministic per-epoch batch order (resume skips consumed batches), wall-clock budget summed across sessions, same W&B run; resumed twice. Kill-and-resume was tested before the real run and caught a bug (RNG state moved to GPU by <code>map_location</code>)</td></tr>
<tr><td>T4 slower than planned (11.5k tok/s)</td><td>Switched 6/6 → 6/3 by a threshold set before measuring</td></tr>
<tr><td>Noisy data; curly vs straight quotes (one ’ costs ~20 chrF on a sentence)</td><td>Filtering (7.7% removed) and one normaliser applied to inputs and outputs</td></tr>
<tr><td>Decoding stalls: one 300-word sentence kept 60 × 4 beams decoding to the length cap</td><td>Length-sorted batches capped at 2,048 source tokens (outputs unchanged)</td></tr>
<tr><td>Tokenizer not reproducible across machines (12 of 16k pieces differ, 2 vs 8 threads)</td><td>Thread count pinned; the trained tokenizer ships with the model</td></tr>
<tr><td>Beam-search edge cases; abbreviation splitting ("M. Dupont")</td><td>Unit tests: greedy = beam(1), overfit-and-recover; abbreviation-aware splitter</td></tr></table>

<h2>8. With more time or compute</h2>
<ul><li>Train to convergence (dev still rising at 9k steps) with a cosine tail; several seeds for variance.</li>
<li>Back-translate monolingual English literature (not opus_books) [12]; domain tags; a copy mechanism for names.</li>
<li>Run the prepared RoPE vs sinusoidal and 16k vs 32k ablations at matched steps; language-ID and dual cross-entropy data filtering [13].</li>
<li>KV cache + MBR decoding with chrF as utility; report sacreBLEU alongside the challenge scorer.</li></ul>

<h2>References</h2>
<div class=refs>
<p>[1] Vaswani et al. 2017, Attention Is All You Need.</p><p>[2] Kasai et al. 2021, Deep Encoder, Shallow Decoder.</p>
<p>[3] Xiong et al. 2020, On Layer Normalization in the Transformer Architecture.</p><p>[4] Su et al. 2021, RoFormer: Rotary Position Embedding.</p>
<p>[5] Press et al. 2022, Train Short, Test Long (ALiBi).</p><p>[6] Press &amp; Wolf 2017, Using the Output Embedding to Improve Language Models.</p>
<p>[7] Kudo &amp; Richardson 2018, SentencePiece.</p><p>[8] Kudo 2018, Subword Regularization.</p>
<p>[9] Sennrich et al. 2016, Neural MT of Rare Words with Subword Units.</p><p>[10] Wu et al. 2016, Google's Neural Machine Translation System.</p>
<p>[11] Popel &amp; Bojar 2018, Training Tips for the Transformer Model.</p><p>[12] Sennrich et al. 2016, Improving NMT Models with Monolingual Data.</p>
<p>[13] Junczys-Dowmunt 2018, Dual Conditional Cross-Entropy Filtering.</p></div>
</body></html>"""


def build_appendix(run, validations):
    """Everything beyond the brief, and every validation with its evidence."""
    rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in validations)
    an = json.load(open(f"{run}/analysis/analysis.json"))
    ex = open(f"{run}/analysis/examples.md").read()
    ex_html = ""
    for block in ex.split("## ")[1:]:
        title, *items = block.strip().split("\n- ")
        ex_html += f"<h3>{esc(title)}</h3><ul>"
        for it in items:
            lines = [x.strip() for x in it.split("\n")]
            head_ = lines[0].replace("**", "").replace("`", "").strip()        # e.g. "6 dev_00055"
            score_, _, sid = head_.partition(" ")
            ex_html += (f"<li><b class=k>chrF {esc(score_)}</b> {esc(sid)}<br>"
                        + "<br>".join(esc(x.lstrip('- ')) for x in lines[1:]) + "</li>")
        ex_html += "</ul>"
    bl = lambda name: "".join(f"<tr><td>{r['bucket']}</td><td class=n>{r['n']}</td><td class=n>{f1(r['bleu'])}</td><td class=n>{f1(r['chrf'])}</td></tr>"
                              for r in an["by_length"][name])
    br = lambda name: "".join(f"<tr><td>{r['bucket']}</td><td class=n>{r['n']}</td><td class=n>{f1(r['bleu'])}</td><td class=n>{f1(r['chrf'])}</td></tr>"
                              for r in an["by_rare"][name])
    head = "<tr><th>bucket</th><th class=n>n</th><th class=n>BLEU</th><th class=n>chrF</th></tr>"
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Appendix: validations and extra work</title>
<style>{CSS} h3 {{ font-size: 9pt; margin: 6px 0 2px; }}</style></head><body>
<h1>Appendix: validations and work beyond the brief</h1>
<div class=links>Companion to report.pdf (3 pages). Every number here comes from files in the repo or the HF model repo
(<code>eval/</code>, <code>analysis/</code>), regenerated by <code>python report/build_report.py</code>.</div>
<h2>A. Work beyond the brief</h2>
<table><tr><th style='width:30%'>Item</th><th>What it adds</th></tr>
<tr><td>Generalization analysis on 2 x 1,000 eval-only sentences</td><td>The 60-sentence dev slices cannot show the gap is real (CI includes zero); the larger samples, length and rare-word buckets and reference loss separate length, vocabulary and register (report section 5, section C below)</td></tr>
<tr><td>Bootstrap confidence intervals</td><td>Per dev slice and for the seen minus unseen gap</td></tr>
<tr><td>Checkpoint-set and decoding studies</td><td>3 checkpoint sets and an 11-point beam x length-penalty grid, extended when the optimum sat at the grid edge</td></tr>
<tr><td>Resumable training for free Colab</td><td>Atomic full-state checkpoints, deterministic batch order, wall-clock budget across sessions; survived two interruptions</td></tr>
<tr><td>Self-contained Hugging Face package</td><td>Weights (safetensors), tokenizer, inference code, model card with real metrics; loads with 3 dependencies</td></tr>
<tr><td>W&amp;B report</td><td>Loss, LR, throughput, dev BLEU/chrF per slice, sample translations, config (scripts/wandb_report.py)</td></tr>
<tr><td>Colab notebook + setup guide</td><td>Secrets handling, Drive checkpoints, throughput check, analysis and Hub push cells (docs/COLAB_SETUP.md)</td></tr>
<tr><td>Reproducible reporting</td><td>report.pdf and this appendix are generated from result files by one script</td></tr>
<tr><td>Prepared ablations</td><td>RoPE vs sinusoidal and 6/6 configs ready to run (configs/)</td></tr></table>
<h2>B. Validations performed, with evidence</h2>
<table><tr><th style='width:22%'>Check</th><th style='width:48%'>What was done</th><th>Result</th></tr>{rows}</table>
<h2>C. Score by source length and by rare-word share (eval-only sets, n=1,000 each)</h2>
<div class=two><div><b class=k>opus-100 test, by length (words)</b><table>{head}{bl('opus100_test')}</table>
<b class=k>opus_books sample, by length</b><table>{head}{bl('opus_books_sample')}</table></div>
<div><b class=k>opus-100 test, by share of rare source words</b><table>{head}{br('opus100_test')}</table>
<b class=k>opus_books sample, by share of rare source words</b><table>{head}{br('opus_books_sample')}</table></div></div>
<h2>D. Lowest- and highest-scoring dev sentences per slice (sentence chrF)</h2>
<div style='font-size:7.6pt'>{ex_html}</div>
</body></html>"""


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
