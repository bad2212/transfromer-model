#!/usr/bin/env bash
# One command, seeded, end to end:
#   bash scripts/reproduce.sh [configs/base_6x3.yaml] [out_dir]
# data -> tokenizer -> train (resumable; just re-run after a disconnect) -> avg ckpts
# -> tune decoding on dev -> dev report + test_predictions.json
set -euo pipefail
CONFIG=${1:-configs/base_6x3.yaml}
OUT=${2:-$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['out_dir'])")}
WORK=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['work_dir'])")

python tests/test_sanity.py
[ -f "$WORK/spm.model" ] || python -m src.prepare_data --config "$CONFIG"
python -m src.train --config "$CONFIG" --out_dir "$OUT"
python -m src.predict --config "$CONFIG" --out_dir "$OUT" --avg 3 --sweep
python score.py --gold data/dev/labels.jsonl --pred "$OUT/eval/dev_predictions.json"
cp "$OUT/eval/test_predictions.json" test_predictions.json
echo "wrote test_predictions.json"
