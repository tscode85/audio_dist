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
2. PANN (Cnn14) weights for FAD via frechet_audio_distance — the Zenodo
   checkpoint is downloaded into ``<cache-dir>/fad_ckpt``.

Usage
-----
    python setup/download_models.py --cache-dir ./model_cache \
        --wavlm microsoft/wavlm-base-plus-sv --fad-model pann --fad-sr 16000

Then on the air-gapped host, set in your config:
    features.wavlm_local_dir:  <cache-dir>/wavlm-base-plus-sv
    features.panns_checkpoint: <cache-dir>/fad_ckpt
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


def download_fad_model(model_name: str, sample_rate: int, ckpt_dir: Path) -> None:
    """Vendor the FAD backbone checkpoint (PANN Cnn14) into ``ckpt_dir``.

    Constructing ``FrechetAudioDistance`` with ``ckpt_dir`` set triggers the
    one-time Zenodo download of the matching Cnn14 weight (e.g.
    ``Cnn14_16k_mAP=0.438.pth`` for 16 kHz). At runtime the same folder is read
    locally with no network.
    """
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except Exception as exc:  # pragma: no cover
        logger.warning("frechet_audio_distance not installed (%s); skipping.", exc)
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Vendoring FAD model '%s' (sr=%d) into %s ...",
                model_name, sample_rate, ckpt_dir)
    FrechetAudioDistance(
        ckpt_dir=str(ckpt_dir),
        model_name=model_name,
        sample_rate=sample_rate,
        use_pca=False,
        use_activation=False,
        verbose=True,
    )
    logger.info("FAD model '%s' cached under %s", model_name, ckpt_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="./model_cache")
    ap.add_argument("--wavlm", default="microsoft/wavlm-base-plus-sv")
    ap.add_argument("--fad-model", default="pann", choices=["pann", "vggish"])
    ap.add_argument("--fad-sr", type=int, default=16000,
                    help="Sample rate; selects the PANN Cnn14 variant (8k/16k/32k).")
    ap.add_argument("--skip-wavlm", action="store_true")
    ap.add_argument("--skip-fad", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    # Keep an HF_HOME inside the cache so it's easy to ship as one directory.
    os.environ.setdefault("HF_HOME", str(cache / "hf"))

    if not args.skip_wavlm:
        wavlm_dir = cache / args.wavlm.split("/")[-1]
        download_wavlm(args.wavlm, wavlm_dir)
    if not args.skip_fad:
        download_fad_model(args.fad_model, args.fad_sr, cache / "fad_ckpt")

    logger.info("Done. Ship '%s' to the air-gapped host and set HF_HUB_OFFLINE=1.", cache)


if __name__ == "__main__":
    main()
