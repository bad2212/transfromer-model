"""Validations performed during the project: (check, what was done, result). Rendered into appendix.pdf."""

VALIDATIONS = [
    ("Model correctness (unit tests)",
     "tests/test_sanity.py: (1) changing a later target token never changes earlier logits (causal mask); "
     "(2) appending padding to the source never changes the output, for RoPE and sinusoidal; (3) a tiny model "
     "memorises 8 random pairs, then greedy reproduces them, beam(k=1, alpha=0) equals greedy exactly, and beam(k=4) "
     "recovers them; (4) normalisation of curly quotes, guillemets, NBSP and ellipsis; (5) sentence splitter incl. "
     "abbreviations such as 'M.'",
     "5/5 pass on Mac (CPU) and on the Colab T4 before training"),
    ("Metric fidelity",
     "src/metrics.py imports the official score.py functions instead of re-implementing them; its OVERALL was "
     "compared with running score.py on the same prediction file",
     "Identical: 7.11 (smoke model) and 39.94 (final model)"),
    ("Data pipeline reproducibility",
     "prepare_data.py run independently on the Mac and on Colab from the same seed",
     "Identical filter counts on both: 923,274 of 1,000,000 kept, every drop reason equal"),
    ("Tokenizer reproducibility",
     "Compared the SentencePiece model rebuilt on the Mac (8 threads) with the one trained on Colab (2 threads)",
     "Not identical: 12 of 16,000 pieces differ, which shifts piece ids. Fixed by pinning 2 threads; the "
     "trained tokenizer ships with every checkpoint and the HF model"),
    ("Train/eval contamination",
     "Exact-match check of dev and test sources against the training sources, before and after normalisation",
     "0 dev and 1 test source in raw train; 3 pairs after normalisation, all removed before training"),
    ("Data-use audit (opus_books)",
     "grep of every dataset load in src/: opus_books appears only in src/analyze.py, run after all model and "
     "decoding choices were fixed; dev/test sentences removed from the sample",
     "Never used for training, tokenizer, checkpoint choice or decoding tuning"),
    ("Resume correctness",
     "Planned kill-and-resume test before the real run: smoke run stopped at step 600, resumed to 800",
     "Caught a real bug (RNG state moved to GPU by map_location; fixed). After the fix: resumed at the exact "
     "batch, loss continued (5.82 -> 5.71 -> 5.60), LR schedule and hours carried over"),
    ("Resume in production",
     "Main run interrupted twice on Colab (disconnect at step ~2,200; GPU loss at step 9,000)",
     "Both resumed from the last checkpoint into the same W&B run (5ks4crpz); no data repeated or skipped"),
    ("Throughput-based model choice",
     "300-step benchmark of the 6/6 model on the T4 against a threshold fixed before measuring (12k target tok/s)",
     "11.5k tok/s, so switched to 6/3 (13.6k tok/s measured in the main run)"),
    ("Training health",
     "Validation loss (opus-100 validation, never used for selection) and dev scores every 2,000 steps",
     "Val loss 4.11 -> 2.65 -> 2.28 -> 2.14 while dev OVERALL rose 14.5 -> 30.8 -> 35.4 -> 37.6: no overfitting; still improving"),
    ("Checkpoint selection",
     "Greedy dev OVERALL for step 9k alone vs averages of 7k-9k and 5k-9k",
     "38.14 / 38.49 / 37.66: averaging the last 3 helps; including early checkpoints hurts. Chose 7k-9k"),
    ("Decoding selection",
     "Grid on dev: greedy; beam 4 and 5 x alpha 0.6/1.0/1.4; extended to alpha 1.8/2.2 because the best value sat at "
     "the grid edge (beam 8 not run: memory pressure, and 4 -> 5 gained < 0.4)",
     "Greedy 38.49 -> beam 5, alpha 1.8: 39.94 (peak; alpha 2.2 drops to 39.83). Gains above alpha 1.4 are within noise"),
    ("Statistical uncertainty",
     "1,000-resample bootstrap 95% CIs per dev slice, and for the seen minus unseen_domain gap",
     "Dev gap 7.9 BLEU, CI [-1.8, 17.5]: not significant on 60 sentences, so the gap was re-measured on "
     "1,000 sentences per domain (31.6 vs 15.8 BLEU, greedy)"),
    ("Submission file",
     "test_predictions.json checked against sample_submission.json",
     "All 330 ids present, none extra, 0 empty, 0 curly quotes"),
    ("Long-input policy",
     "Splitting only triggers above 128 subword tokens (never on dev); compared split vs no split on long eval-only sentences",
     "23 such sentences: chrF 41.2 split vs 39.5 not split (BLEU 19.6 vs 19.0); kept as the default"),
    ("Failure-mode claims",
     "Each failure claim in the report checked against the 2,000 eval-only outputs before writing it: repetition counted "
     "(a 3-gram repeated 3+ times); literary past tenses inspected; rare-word buckets made disjoint after finding a "
     "boundary double count",
     "Repetition 0.5% / 0.4% of outputs (in-domain / books); an over-strong 'tense flattened' claim and a wrong "
     "'both domains fall with rare words' caption were corrected before publishing"),
    ("Hugging Face model integrity",
     "SHA-256 of model.safetensors and spm.model on the Hub compared with the local export; sizes for all other files",
     "All 12 files match"),
    ("Hugging Face model usability",
     "The exported package loaded with only torch, sentencepiece and safetensors (no repo code) via "
     "fr_en_transformer.load_translator, as in the model card, and run on CPU",
     "Loads and translates 3 sentences in 0.2 s, e.g. 'Ne t'inquiete pas !' -> 'Don't worry!'"),
    ("Secrets and confidentiality",
     "Every commit scanned for the HF and W&B key patterns; the brief (PDF) and plan are git-ignored; keys only in "
     "a local .env and Colab Secrets; a repo-only deploy key for GitHub",
     "0 secrets in the public repo"),
]
