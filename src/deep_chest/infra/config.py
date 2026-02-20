import mlflow
import os
from pathlib import Path





def init_mlflow(exp_name: str):
    """
    Initialize MLflow for both local and Colab runs.
    - Local: uses sqlite DB + local mlruns folder
    - Colab: uses mlruns folder inside repo
    """
    # detect if running in Colab
    IN_COLAB = "COLAB_GPU" in os.environ

    if IN_COLAB:
        # Use mlruns folder inside repo
        repo_root = Path(__file__).resolve().parents[3]  # adjust to repo root
        artifact_dir = repo_root / "mlruns"
        artifact_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{artifact_dir}")
    else:
        # Local machine: sqlite + mlruns
        repo_root = Path(__file__).resolve().parents[3]
        mlflow.set_tracking_uri(f"sqlite:///{repo_root / 'mlflow.db'}")
        artifact_dir = repo_root / "mlruns"
        artifact_dir.mkdir(exist_ok=True)

    # Set experiment
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