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