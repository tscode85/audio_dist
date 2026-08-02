# file: setup/download_models.py
"""One-time, network-connected model pre-download for AIR-GAPPED deployment.

Run this ONCE on a machine WITH internet access to vendor every checkpoint the
toolkit needs into a local cache directory. Copy that directory to the air-gapped
host and point the toolkit at it via config / environment variables. At runtime,
``HF_HUB_OFFLINE=1`` and ``local_files_only=True`` guarantee nothing hits the net.

What it fetches
---------------
1. WavLM x-vector model (HuggingFace):  microsoft/wavlm-base-plus-sv
   (or microsoft/wavlm-base-sv) — feature extractor + weights.
2. PANN / VGGish weights for FAD via fadtk (downloaded through fadtk's own
   downloader) or frechet_audio_distance.

Usage
-----
    python setup/download_models.py --cache-dir ./model_cache \
        --wavlm microsoft/wavlm-base-plus-sv --fadtk-model vggish

Then on the air-gapped host, set in your config:
    features.wavlm_local_dir: <cache-dir>/wavlm-base-plus-sv
and export:
    export HF_HOME=<cache-dir>/hf
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("download_models")


def download_wavlm(model_name: str, out_dir: Path) -> None:
    """Snapshot a WavLM x-vector model + feature extractor to ``out_dir``."""
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", model_name)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name)
    extractor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    logger.info("Saved WavLM snapshot to %s", out_dir)


def download_fadtk_model(model_name: str) -> None:
    """Trigger fadtk to fetch/cache PANN/VGGish weights locally."""
    try:
        from fadtk.model_loader import get_all_models
    except Exception as exc:  # pragma: no cover
        logger.warning("fadtk not installed (%s); skipping FAD weights.", exc)
        return
    models = {m.name: m for m in get_all_models()}
    if model_name not in models:
        logger.warning("fadtk model '%s' not found; available: %s",
                       model_name, sorted(models))
        return
    logger.info("Loading fadtk model '%s' to populate its cache ...", model_name)
    models[model_name].load_model()
    logger.info("fadtk model '%s' cached.", model_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="./model_cache")
    ap.add_argument("--wavlm", default="microsoft/wavlm-base-plus-sv")
    ap.add_argument("--fadtk-model", default="vggish")  # fadtk==1.0.0 provides vggish
    ap.add_argument("--skip-wavlm", action="store_true")
    ap.add_argument("--skip-fadtk", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    # Keep an HF_HOME inside the cache so it's easy to ship as one directory.
    os.environ.setdefault("HF_HOME", str(cache / "hf"))

    if not args.skip_wavlm:
        wavlm_dir = cache / args.wavlm.split("/")[-1]
        download_wavlm(args.wavlm, wavlm_dir)
    if not args.skip_fadtk:
        download_fadtk_model(args.fadtk_model)

    logger.info("Done. Ship '%s' to the air-gapped host and set HF_HUB_OFFLINE=1.", cache)


if __name__ == "__main__":
    main()
