# French → English Transformer, trained from scratch

Encoder-decoder Transformer (pre-LN, RoPE, tied embeddings, joint SentencePiece) trained from
scratch on a filtered subset of `Helsinki-NLP/opus-100` (en-fr), evaluated on in-domain, long, and
unseen-domain (`opus_books`, never trained on) slices.

| | Link |
|---|---|
| Model (Hugging Face) | https://huggingface.co/Badalt/fr-en-transformer-scratch |
| Training run (W&B) | https://wandb.ai/badalthakur2212-iisc/fr-en-transformer/runs/5ks4crpz |
| W&B report | [training and generalization](https://wandb.ai/badalthakur2212-iisc/fr-en-transformer/reports/FR%E2%86%92EN-Transformer-from-scratch:-training-and-generalization--VmlldzoxODA1MDg0Mw==) |
| Report | [`report/report.pdf`](report/report.pdf) (built by `python report/build_report.py`) |
| Submission | [`test_predictions.json`](test_predictions.json) |

## Results (provided dev set, official `score.py`)

6 encoder / 3 decoder layers, 39.7M parameters, 3.6 h on one T4 (9,000 updates, ~8.3 epochs; stopped by the
free-Colab GPU quota), average of checkpoints 7k-9k, beam 5, length penalty 1.8.

| Slice | n | BLEU | chrF |
|---|---|---|---|
| seen | 60 | 29.78 | 46.37 |
| long | 30 | 34.66 | 59.98 |
| unseen_domain | 60 | 21.89 | 43.41 |
| **all** | 150 | **30.23** | **47.91** |

**OVERALL = 39.94** (0.4·BLEU + 0.4·chrF + 0.2·chrF on unseen_domain). Length is not the failure mode at these
lengths; domain is (see the report).

## Reproduce (one command, seeded)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export WANDB_API_KEY=...            # or set wandb.enabled: false in the config
bash scripts/reproduce.sh configs/base_6x3.yaml   # the submitted model
```

Steps: sanity tests → data prep + tokenizer (`src/prepare_data.py`) → training (`src/train.py`,
resumable: re-run the same command after an interruption) → checkpoint averaging + beam/length-penalty
sweep on dev + test predictions (`src/predict.py`) → `score.py` on dev → `test_predictions.json`.

On Colab (T4), follow [docs/COLAB_SETUP.md](docs/COLAB_SETUP.md) (`notebooks/colab_train.ipynb`); checkpoints go to Google Drive so a disconnect only
costs a re-run of the training cell.

Local smoke test (CPU/Apple-GPU, ~10 min, not for results): `bash scripts/reproduce.sh configs/smoke.yaml`

## Layout

| Path | What |
|---|---|
| `configs/` | `base_6x3.yaml` (submitted model), `base.yaml` (6/6, benchmarked), `ablation_sinusoidal.yaml`, `smoke.yaml` |
| `src/prepare_data.py` | load opus-100, normalise, filter, dedupe, drop dev/test overlap, train SentencePiece |
| `src/model.py` | the Transformer (written from scratch on `torch.nn` primitives + `scaled_dot_product_attention`) |
| `src/train.py` | token-bucketed batches, source-side subword sampling, AMP, warmup + inverse-sqrt LR, W&B, resumable checkpoints |
| `src/decode.py` | greedy, beam search with GNMT length penalty, long-input splitting, output normalisation |
| `src/predict.py` | checkpoint averaging, dev sweep, dev/test predictions |
| `src/metrics.py` | dev scoring via the official `score.py` functions |
| `tests/test_sanity.py` | causality, padding invariance, overfit-and-decode, normalisation, splitting |
| `score.py` | official scorer (unchanged) |
| `data/dev`, `data/test` | frozen eval sets (provided) |

## Data

- Train: `Helsinki-NLP/opus-100`, config `en-fr`, `train` split, fr → en. Validation split for loss.
- Unseen-domain eval: `Helsinki-NLP/opus_books` (`en-fr`). **Not used for training or model selection.**
- Submission format: `{"<id>": "<english translation>", ...}`; score with
  `python3 score.py --gold data/dev/labels.jsonl --pred <file>.json`.
