#!/usr/bin/env bash
# Rebuild the conda environment `neurofly` on x86_64 Linux. Usage: bash env/create_neurofly.sh [path/to/conda]
set -euo pipefail
CONDA="${1:-$HOME/miniforge3/bin/conda}"
HERE="$(cd "$(dirname "$0")" && pwd)"
"$CONDA" env create -n neurofly -f "$HERE/neurofly_x86_64.yml"
PY="$("$CONDA" run -n neurofly python -c 'import sys; print(sys.executable)')"
env -u PYTHONPATH "$PY" -m pip install --no-deps -r "$HERE/neurofly_x86_64_nodeps.txt"
env -u PYTHONPATH "$PY" -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
