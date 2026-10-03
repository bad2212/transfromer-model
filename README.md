# French → English Transformer, trained from scratch

Encoder-decoder Transformer (pre-LN, RoPE, tied embeddings, joint SentencePiece) trained from
scratch on a filtered subset of `Helsinki-NLP/opus-100` (en-fr), evaluated on in-domain, long, and
unseen-domain (`opus_books`, never trained on) slices.

| | Link |
|---|---|
| Model (Hugging Face) | _TBD_ |
| Training run (W&B) | _TBD_ |
| Report | `report/report.pdf` (_TBD_) |

## Reproduce (one command, seeded)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export WANDB_API_KEY=...            # or set wandb.enabled: false in the config
bash scripts/reproduce.sh configs/base.yaml
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
| `configs/` | `base.yaml` (main run), `base_6x3.yaml` (fallback), `ablation_sinusoidal.yaml`, `smoke.yaml` |
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
