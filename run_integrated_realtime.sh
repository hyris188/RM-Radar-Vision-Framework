#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rm-radar-mpl}"
export LD_LIBRARY_PATH="/opt/MVS/lib/64:${LD_LIBRARY_PATH:-}"

exec python3 -m deployment.integrated_inference "$@"
