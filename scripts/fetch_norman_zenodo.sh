#!/bin/bash
# Zenodo throttles this box to ~30-190 KB/s and drops the connection every few minutes,
# so pull the full Norman2019 atlas with a resume loop instead of a single curl.
# Runs unattended; prep falls back to the HuggingFace subsample until this lands.
set -u
OUT="${1:-/data/projects/yyc/datasets/norman/NormanWeissman2019_filtered.h5ad}"
URL="https://zenodo.org/records/13350497/files/NormanWeissman2019_filtered.h5ad?download=1"
SIZE=698680199
for i in $(seq 1 400); do
  have=$(stat -c%s "$OUT.part" 2>/dev/null || echo 0)
  if [ "$have" -ge "$SIZE" ]; then mv "$OUT.part" "$OUT"; echo "NORMAN_FULL_DONE $OUT"; exit 0; fi
  echo "[try $i] have $have / $SIZE bytes"
  curl -sSL --fail -C - --max-time 1800 --retry 3 --retry-all-errors -o "$OUT.part" "$URL" || true
  sleep 5
done
echo "NORMAN_FULL_INCOMPLETE"; exit 1
