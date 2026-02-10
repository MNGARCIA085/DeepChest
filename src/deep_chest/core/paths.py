from pathlib import Path
import os

def get_data_root() -> Path:
    try:
        return Path(os.environ["DATA_ROOT"]).resolve()
    except KeyError:
        raise RuntimeError(
            "DATA_ROOT is not set. "
            "Set it via environment variable."
        )





# in console: export DATA_ROOT=/home/marcos/Escritorio/AI-prod/DeepChest/data/nih



"""
from pathlib import Path

# src/deep_chest/core/paths.py

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACKING_DIR = PROJECT_ROOT / "tracking"
LEADERBOARD_PATH = TRACKING_DIR / "leaderboard.json"
"""