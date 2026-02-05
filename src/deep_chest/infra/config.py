import mlflow
import os
from pathlib import Path



def init_mlflow(exp_name):
    # Project root (3 levels up from this file)
    root_dir = Path(__file__).resolve().parents[3]

    # Tracking DB
    mlflow.set_tracking_uri(f"sqlite:///{root_dir / 'mlflow.db'}")

    # Artifacts folder
    artifact_dir = root_dir / "mlruns"
    os.makedirs(artifact_dir, exist_ok=True)

    # set experimnet name
    mlflow.set_experiment(exp_name)