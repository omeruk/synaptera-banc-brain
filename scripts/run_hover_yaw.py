"""Run a v1 hover + yaw-perturbation reproduction (tasks/hover_yaw.py); under nohup, results in runs/hover_yaw/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.hover_yaw import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
