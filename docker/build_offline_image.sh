#!/usr/bin/env bash
# file: docker/build_offline_image.sh
# Build the offline GPU image (WITH internet) and export it as a portable
# tarball to carry onto the air-gapped host.
#
# Run from the repository ROOT:
#   ./docker/build_offline_image.sh
#
# Env overrides:
#   IMAGE      image tag            (default: acoustic-gap:offline)
#   OUTPUT     output tar.gz path   (default: acoustic-gap-offline.tar.gz)
#   WAVLM      WavLM model id       (default: microsoft/wavlm-base-plus-sv)
#   FADTK      fadtk model name     (default: vggish; fadtk==1.0.0 has no PANN)
#   DOWNLOAD_MODELS  1|0            (default: 1 — bake weights into the image)
set -euo pipefail

IMAGE="${IMAGE:-acoustic-gap:offline}"
OUTPUT="${OUTPUT:-acoustic-gap-offline.tar.gz}"
WAVLM="${WAVLM:-microsoft/wavlm-base-plus-sv}"
FADTK="${FADTK:-vggish}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">> Building $IMAGE (bake weights: DOWNLOAD_MODELS=$DOWNLOAD_MODELS) ..."
docker build \
    -f docker/Dockerfile \
    --build-arg DOWNLOAD_MODELS="$DOWNLOAD_MODELS" \
    --build-arg WAVLM_MODEL="$WAVLM" \
    --build-arg FADTK_MODEL="$FADTK" \
    -t "$IMAGE" \
    .

echo ">> Exporting $IMAGE -> $OUTPUT ..."
docker save "$IMAGE" | gzip -c > "$OUTPUT"

echo ">> Done."
echo "   Copy '$OUTPUT' to the air-gapped host, then:"
echo "     docker load -i $OUTPUT"
echo "     docker run --rm --gpus all \\"
echo "        -v /path/to/real:/data/real:ro \\"
echo "        -v /path/to/sim:/data/sim:ro \\"
echo "        -v /path/to/report:/data/report \\"
echo "        $IMAGE run --config /app/config/offline_gpu.yaml \\"
echo "        --real /data/real --sim /data/sim --output-dir /data/report"
