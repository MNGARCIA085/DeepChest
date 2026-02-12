import mlflow
import json
from mlflow.tracking import MlflowClient
from pathlib import Path
from dataclasses import dataclass
from keras.models import load_model



client = MlflowClient()
experiment = client.get_experiment_by_name('deep_chest') # later pass it from config


#------------------------Load data for inference--------------------------#
@dataclass
class InferenceBundle:
    model: object
    labels: dict | list
    model_type: str
    prep_fn: str


def load_inference_bundle(run_id: str, dst="artifacts"):
    client = MlflowClient()

    # --- RUN DATA ---
    run = client.get_run(run_id)

    model_type = run.data.tags.get("model_type")
    prep_fn = run.data.params.get("preprocessing")

    # --- ARTIFACTS ---
    artifacts_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/",
        dst_path=dst
    )

    # 1. MODEL
    model = load_model(f"{artifacts_path}/model/model.keras", compile=False)

    # check later if this works; use mlflow.keras instead of mlflow.tensorflow
    #model_uri = f"runs:/{run_id}/model"
    # model = mlflow.keras.load_model(model_uri)

    # 2. LABELS
    labels_local = client.download_artifacts(
        run_id,
        "labels.json",
        str(artifacts_path)
    )

    with open(labels_local, "r") as f:
        labels = json.load(f)


    return InferenceBundle(
        model=model,
        labels=labels,
        model_type=model_type,
        prep_fn=prep_fn,
    )