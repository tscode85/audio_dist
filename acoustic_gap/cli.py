# file: acoustic_gap/cli.py
"""Command-line entry point for the acoustic-gap toolkit.

Examples
--------
Run an end-to-end comparison from a YAML config::

    python -m acoustic_gap.cli run --config config/default.yaml \
        --real /data/real --sim /data/sim --output-dir ./report

Emit a starter config::

    python -m acoustic_gap.cli init-config --output config/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import AppConfig
from .logging_utils import configure_logging
from .pipeline import run_pipeline

logger = logging.getLogger("acoustic_gap")


def _cmd_run(args: argparse.Namespace) -> int:
    if args.config:
        cfg = AppConfig.load(args.config)
    else:
        cfg = AppConfig()
    # CLI overrides.
    if args.real:
        cfg.real_dir = args.real
    if args.sim:
        cfg.sim_dir = args.sim
    if args.output_dir:
        cfg.report.output_dir = args.output_dir
    if args.backbones:
        cfg.features.backbones = args.backbones
    if args.log_level:
        cfg.log_level = args.log_level
    cfg.validate()
    configure_logging(cfg.log_level)

    if not cfg.real_dir or not cfg.sim_dir:
        logger.error("Both --real and --sim (or config real_dir/sim_dir) are required.")
        return 2

    report = run_pipeline(cfg)
    contributor = report["headline"]["largest_contributor"]
    print("\n=== Acoustic Domain Gap Summary ===")
    print(report["interpretation"])
    print("\nTop-level scores (lower = smaller gap):")
    for backbone, metrics in report["top_level_scores"].items():
        for metric, value in metrics.items():
            print(f"  {backbone:8s} {metric:16s} {value}")
    if contributor:
        print(
            f"\n>>> Largest domain-gap contributor: '{contributor['condition']}' "
            f"({contributor['backbone']}/{contributor['metric']} = "
            f"{contributor['value']:.4f})"
        )
    print(f"\nFull report written under: {cfg.report.output_dir}")
    return 0


def _cmd_init_config(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(out)
    print(f"Wrote starter config to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acoustic_gap",
        description="Quantify the acoustic domain gap between real and simulated audio.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an end-to-end real-vs-sim comparison.")
    run.add_argument("--config", help="Path to YAML/JSON config.")
    run.add_argument("--real", help="Directory of real audio.")
    run.add_argument("--sim", help="Directory of simulated audio.")
    run.add_argument("--output-dir", help="Report output directory.")
    run.add_argument(
        "--backbones", nargs="+", choices=["panns", "vggish", "wavlm", "dummy"],
        help="Override the configured feature backbones.",
    )
    run.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR.")
    run.set_defaults(func=_cmd_run)

    init = sub.add_parser("init-config", help="Write a default config file.")
    init.add_argument("--output", default="config/default.yaml", help="Output path.")
    init.set_defaults(func=_cmd_init_config)

    return p


def main(argv=None) -> int:
    configure_logging("INFO")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
