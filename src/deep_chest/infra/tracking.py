import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

client = MlflowClient()
experiment = client.get_experiment_by_name('deep_chest') # later pass it


from .utils import get_best_run_id, get_top_k_run_ids



# see later, i log as an artifact bc version conflicts
def download_best_model(dst="downloaded_model"):
    run_id = get_best_run_id()
    print(run_id)
    if not run_id:
        return None

    uri = f"runs:/{run_id}/model"
    return mlflow.artifacts.download_artifacts(uri, dst_path=dst)



def download_all_artifacts(run_id, dst="artifacts"):
    return mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/",
        dst_path=dst
    )


def download_top_k_models(k=2):
    paths = []
    for i, run_id in enumerate(get_top_k_run_ids(k)):
        dst = f"best_model_{i}"
        uri = f"runs:/{run_id}/model"
        path = mlflow.artifacts.download_artifacts(uri, dst_path=dst)
        paths.append(path)
    return paths
