"""Create (or re-create) the W&B report for the training run(s). Panels are live: they keep
updating while the run trains.

    WANDB_API_KEY=... python scripts/wandb_report.py --entity badalthakur2212-iisc --project fr-en-transformer
"""
import argparse

import wandb_workspaces.reports.v2 as wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", default="fr-en-transformer")
    ap.add_argument("--github", default="https://github.com/bad2212/transfromer-model")
    a = ap.parse_args()

    runs = wr.Runset(entity=a.entity, project=a.project, name="training runs")
    slices = ["seen", "long", "unseen_domain"]
    report = wr.Report(
        entity=a.entity, project=a.project,
        title="FR→EN Transformer from scratch: training and generalization",
        description="Loss, learning rate, throughput and dev BLEU/chrF by slice for the from-scratch fr→en model.",
        blocks=[
            wr.TableOfContents(),
            wr.H1("Overview"),
            wr.P("Encoder-decoder Transformer (pre-LN, RoPE, tied embeddings, 16k joint SentencePiece), "
                 "6 encoder / 3 decoder layers, trained from scratch on filtered opus-100 en-fr (923k pairs) "
                 "on a single Colab T4. Dev scores use the official score.py (greedy decoding during training; "
                 "the final numbers use checkpoint averaging and beam search)."),
            wr.P([wr.Link("Code and one-command reproduction (GitHub)", url=a.github)]),
            wr.H1("Training"),
            wr.PanelGrid(runsets=[runs], panels=[
                wr.LinePlot(title="Train loss (label-smoothed CE)", x="Step", y=["train/loss"]),
                wr.LinePlot(title="Validation loss (opus-100 validation)", x="Step", y=["val/loss"]),
                wr.LinePlot(title="Learning rate (warmup + inverse sqrt)", x="Step", y=["train/lr"]),
                wr.LinePlot(title="Gradient norm (before clipping at 1.0)", x="Step", y=["train/grad_norm"]),
                wr.LinePlot(title="Throughput (target tokens/s, T4 fp16)", x="Step", y=["train/tgt_tokens_per_s"]),
                wr.LinePlot(title="Training hours (across resumes)", x="Step", y=["train/hours"]),
            ]),
            wr.H1("Dev scores (official scorer)"),
            wr.P("OVERALL = 0.4·BLEU(all) + 0.4·chrF(all) + 0.2·chrF(unseen_domain). The gap between the seen "
                 "and unseen_domain lines is the generalization gap the task asks about."),
            wr.PanelGrid(runsets=[runs], panels=[
                wr.LinePlot(title="Dev OVERALL", x="Step", y=["dev/overall"]),
                wr.LinePlot(title="Dev BLEU by slice", x="Step", y=[f"dev/{s}/bleu" for s in slices]),
                wr.LinePlot(title="Dev chrF by slice", x="Step", y=[f"dev/{s}/chrf" for s in slices]),
                wr.LinePlot(title="Dev BLEU / chrF (all)", x="Step", y=["dev/bleu", "dev/chrf"]),
            ]),
            wr.H1("Sample translations"),
            wr.P("First 20 dev sentences at each evaluation (source, hypothesis, reference)."),
            wr.PanelGrid(runsets=[runs], panels=[wr.WeavePanelSummaryTable(table_name="dev/samples")]),
            wr.H1("Run config / hyper-parameters"),
            wr.PanelGrid(runsets=[runs], panels=[wr.RunComparer()]),
        ],
    )
    report.save()
    print(report.url)


if __name__ == "__main__":
    main()
