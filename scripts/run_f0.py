"""Run the F0 acceptance flight (see tasks/f0_flight.py). Always under nohup, results under runs/f0/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks.f0_flight import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
