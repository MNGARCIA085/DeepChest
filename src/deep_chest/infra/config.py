from pathlib import Path
import os
import mlflow
import sys

def init_mlflow(exp_name: str):
    # detect Colab
    IN_COLAB = "google.colab" in sys.modules

    print(IN_COLAB)

    if IN_COLAB:
        # Repo root = current working directory in Colab
        repo_root = Path.cwd()
        artifact_dir = repo_root / "mlruns"
        artifact_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{artifact_dir}")
    else:
        # Local machine: same as before
        repo_root = Path(__file__).resolve().parents[3]
        artifact_dir = repo_root / "mlruns"
        artifact_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{repo_root / 'mlflow.db'}")

    mlflow.set_experiment(exp_name)
    print(f"MLflow initialized. URI: {mlflow.get_tracking_uri()}")





"""
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
"""