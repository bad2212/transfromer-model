# Training on Google Colab: setup guide

This guide trains the main model on a free Colab T4 GPU (16 GB). The run takes about 5.5 hours of
GPU time. Checkpoints are saved to Google Drive, so a Colab disconnect costs only a few minutes.

**Before you start**
- The GitHub repo `bad2212/transfromer-model` is **public**. Colab clones it without logging in.
- About **3 GB free on Google Drive** for checkpoints.
- A Hugging Face token with **write** access, and a Weights & Biases API key.

---

## Step 1: Open the notebook in Colab

1. Go to https://colab.research.google.com
2. **File → Open notebook → GitHub** tab → paste `bad2212/transfromer-model` → open
   `notebooks/colab_train.ipynb`.
3. **File → Save a copy in Drive**. Your edits and outputs are then saved to your own copy.

## Step 2: Select a GPU

**Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

## Step 3: Add the Hugging Face and W&B keys as Colab Secrets

Secrets are stored with your Google account and are never saved in the notebook. That keeps them out
of GitHub.

1. Click the **key icon** in the left sidebar.
2. **+ Add new secret**, twice:

   | Name (exactly) | Value |
   |---|---|
   | `HF_TOKEN` | your Hugging Face token (starts with `hf_`). Create one at https://huggingface.co/settings/tokens with **write** permission. |
   | `WANDB_API_KEY` | your W&B API key from https://wandb.ai/authorize |

3. Switch **Notebook access** on for both secrets.

Cell 1 checks both secrets and stops with a clear message if one is missing.

## Step 4: Run the cells in order

| Cell | What it does | Time |
|---|---|---|
| 1 | Mounts Google Drive (approve the popup) and loads the secrets | <1 min |
| 2 | Clones the repo, installs dependencies, prints the GPU name (should say Tesla T4) | ~1 min |
| 3 | Downloads opus-100, filters it, trains the tokenizer, caches everything on Drive | ~10 min the first time, seconds afterwards |
| 4 | Sanity tests + a 300-step **throughput check** with W&B off | ~5 min |
| 5 | **Main training run** with live W&B logging | ~5.5 h |
| 6 | Checkpoint averaging, beam/length-penalty sweep on dev, dev score, `test_predictions.json` | ~15 min |

**After cell 4**, look at `tgt_tokens_per_s` in the last `step 300` line:
- **≥ ~12,000**: keep `CONFIG = 'configs/base.yaml'` in cell 5.
- **< ~12,000**: change cell 5 to `CONFIG = 'configs/base_6x3.yaml'` (same encoder, 3 decoder layers,
  faster). Also change `OUT` in cell 1 to `.../fr-en-transformer/base_6x3`.

Cell 5 prints a **W&B run link** near the top (`wandb: View run at https://wandb.ai/...`). Open it to
watch the loss and dev BLEU/chrF curves. Every 2,000 steps it also prints a line like
`eval step 2000: val_loss=... dev OVERALL=...`.

## Step 5: If Colab disconnects

Free Colab can drop the session after inactivity or after several hours. Nothing is lost:

1. Reconnect (Runtime → Connect, T4 again).
2. Run **cells 1, 2 and 3** (fast, because data is cached), then **cell 5** again.
3. It prints `resumed from step N` and continues. The same W&B run continues too, and the time budget
   carries over.

To reduce disconnects, keep the browser tab open and the laptop awake.

## Step 6: Results

Everything is in Google Drive under `MyDrive/fr-en-transformer/`:

| File | What |
|---|---|
| `base/best.pt`, `base/step_*.pt` | best checkpoint by dev score, the last 5 checkpoints (averaged in cell 6) |
| `base/spm.model` | tokenizer |
| `base/eval/dev_report.json`, `sweep.json` | dev scores per slice, decoding sweep table |
| `test_predictions.json` | the submission file |

Send back `dev_report.json`, `sweep.json` and the W&B link. The analysis, the Hugging Face upload and
the report come next.

## Step 7: Make the W&B project public

On wandb.ai, open the project **fr-en-transformer** → **Settings** (or the lock icon next to the
project name) → **Visibility: Public**. If the project is under a team entity, the team must allow
public projects.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Secret HF_TOKEN missing or notebook access not enabled` | Step 3: check the exact name and that Notebook access is on |
| `git clone` asks for a password | The GitHub repo is private. Make it public (repo Settings → Danger zone → Change visibility). |
| `CUDA out of memory` | In the config, set `train.batch_tokens: 6144` and `accum_steps: 4` (same tokens per update) |
| Cell 2 shows no GPU / `nvidia-smi` fails | Step 2: the runtime isn't set to a GPU, or no free GPU is available right now; try again later |
| Loss becomes `nan` | Report it. Lowering `train.lr` to `0.0005` and resuming from the last checkpoint is the usual fix. |
