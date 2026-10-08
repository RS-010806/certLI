#!/usr/bin/env bash
# Downloads the two BEIR corpora and the two ColBERT checkpoints into assets/.
set -euo pipefail
ROOT="${CERTLI_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/assets"
for ds in scifact nfcorpus; do
  [ -f "$ROOT/assets/beir_${ds}.zip" ] || \
    curl -L -o "$ROOT/assets/beir_${ds}.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${ds}.zip"
done
python - "$ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
root = sys.argv[1]
snapshot_download("colbert-ir/colbertv2.0", local_dir=f"{root}/assets/colbertv2")
snapshot_download("answerdotai/answerai-colbert-small-v1", local_dir=f"{root}/assets/answerai-small")
PY
echo "assets ready in $ROOT/assets"
