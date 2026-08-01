#!/usr/bin/env bash
# file: docker/entrypoint.sh
# Thin entrypoint for the acoustic-gap container.
#   * `run ...`        -> pipeline CLI (default)
#   * `init-config ...`-> emit a starter config
#   * `test`           -> run the offline pytest suite
#   * `bash`/`sh`      -> drop into a shell for debugging
#   * anything else    -> executed verbatim
set -euo pipefail

case "${1:-}" in
  test)
    shift
    exec python -m pytest -q "$@"
    ;;
  bash|sh)
    exec "$@"
    ;;
  ""|run|init-config)
    exec python -m acoustic_gap.cli "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
