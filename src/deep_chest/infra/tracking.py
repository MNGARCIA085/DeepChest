import mlflow
import json
from mlflow.tracking import MlflowClient
from pathlib import Path
from .utils import get_best_run_id, get_top_k_run_ids
from dataclasses import dataclass
from keras.models import load_model



client = MlflowClient()
experiment = client.get_experiment_by_name('deep_chest') # later pass it



# see later, i log as an artifact bc version conflicts
def download_best_model(dst="downloaded_model"):
    run_id = get_best_run_id()
    print(run_id)
    if not run_id:
        return None

    uri = f"runs:/{run_id}/model"
    return mlflow.artifacts.download_artifacts(uri, dst_path=dst)


def download_top_k_models(k=2):
    paths = []
    for i, run_id in enumerate(get_top_k_run_ids(k)):
        dst = f"best_model_{i}"
        uri = f"runs:/{run_id}/model"
        path = mlflow.artifacts.download_artifacts(uri, dst_path=dst)
        paths.append(path)
    return paths




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

    # check later if this works
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