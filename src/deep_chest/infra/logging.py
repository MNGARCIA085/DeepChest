import mlflow
import joblib
import os
import json

from dataclasses import is_dataclass, asdict
#from net_security.utils.plots import plot_cm, plot_roc, plot_train_val

from hydra.core.hydra_config import HydraConfig


# tags
def log_tags(stage, model_type, dataset, extra_tags=None):

    tags = {
        "stage": stage,
        "model_type": model_type,
        "task": "multilabel_classification",
        "dataset": dataset,
        "data_version": "v1",

    }

    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)


# comon params
def log_common_params(artifacts):
    #mlflow.log_param("val_size", artifacts["val_size"])
    #mlflow.log_param("train_shape", artifacts["train_shape"])
    #mlflow.log_param("val_shape", artifacts["val_shape"])
    mlflow.log_param("image_size", artifacts["image_size"])
    # labels as dict? mlflow.log_dict({"features": artifacts["features"]}, "features.json")
    # pos_weights?????



# metrics
def log_metrics(metrics):
    pass


# hyperparams
def log_hyperparamsv0(hyperparams):
    mlflow.log_params(hyperparams)



# training curves
def log_training_curves(train_data, val_data, filename, title):
    loss_path = plot_train_val(train_data, val_data, filename, title)
    mlflow.log_artifact(loss_path)
    os.remove(loss_path)




"""
better versions!!!!!
"""

def log_hyperparams(hyperparams: dict, prefix=""):
    for k, v in hyperparams.items():
        mlflow.log_param(f"{prefix}{k}", v)


# later maybe just curves
def log_history(history: dict, prefix=""):
    for metric_name, values in history.items():
        for epoch, v in enumerate(values):
            mlflow.log_metric(f"{prefix}{metric_name}", v, step=epoch)


# nota -> puse 1 epoch!!!!!!!!


#final metrics after training
def log_final_metrics(history, prefix=""):
    for metric, values in history.items():
        mlflow.log_metric(f"{prefix}final_{metric}", values[-1])
        mlflow.log_metric(f"{prefix}best_{metric}", max(values))



def log_training_results(results):

    if "history" in results:  # Standard
        log_hyperparams(results["hyperparams"])
        #log_final_metrics(results['history'])
        #log_history(results["history"])

    else:  # Transfer learning
        for phase, data in results.items():
            log_hyperparams(
                {k: v for k, v in data.items() if k != "history"},
                prefix=f"{phase}_"
            )
            #log_final_metrics(data["history"], prefix=f"{phase}_")
            #log_history(data["history"], prefix=f"{phase}_")




def log_per_class_metrics(metrics: dict, prefix=""): # prefix=metrics
    for metric_name, arr in metrics.items():
        for i, v in enumerate(arr):
            mlflow.log_metric(
                f"{prefix}_{metric_name}_class_{i}",
                float(v)
            )






#--------------------------------------------------#
import pandas as pd

def build_per_class_df(metrics: dict):
    df = pd.DataFrame(metrics)
    df.index.name = "class_id"
    return df


from hydra.core.hydra_config import HydraConfig
import os

def log_per_class_df(df, filename="per_class_metrics.csv"):
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    df.to_csv(path, index=True)
    mlflow.log_artifact(path)

"""
as json too
json_path = path.replace(".csv", ".json")
df.to_json(json_path, orient="index")
mlflow.log_artifact(json_path)
"""


#---------------------------------------------------#

# metrics aggregates

import numpy as np

def log_metric_means(metrics, prefix=""):
    for name, arr in metrics.items():
        mlflow.log_metric(
            f"{prefix}_{name}_mean",
            float(np.mean(arr))
        )





#-----------------------------------------------#




# log exp. log_Experiment
def logging(run_name, artifacts, results, model_type, stage):

    with mlflow.start_run(run_name=run_name):
        log_tags(stage, model_type, "train/val")
        """
        log_common_params(artifacts)
        log_metrics(results)
        log_hyperparams(results['hyperparams'])
        """
        #log_per_class_metrics(results['metrics'])

        df = build_per_class_df(results['metrics'])
        log_per_class_df(df) # later pus class name instead of class_id

        # log metrics aggregate
        log_metric_means(results['metrics'])

        log_training_results(results)



"""
only for tl
results = trainer.train(...)

for phase_name, phase_data in results.items():

    # ---- params ----
    phase_params = {
        "lr": phase_data["lr"],
        "epochs": phase_data["epochs"],
    }

    if "num_unfrozen" in phase_data:
        phase_params["num_unfrozen"] = phase_data["num_unfrozen"]

    log_params(phase_params, prefix=f"{phase_name}_")

    # ---- history ----
    log_history(phase_data["history"], prefix=f"{phase_name}_")
"""
        






"""
#------------------Final evaluation (with test set)-----------------------------
def log_test_results(tuning_run_id, model_type, results, stage='eval'):
    with mlflow.start_run(run_name="test_evaluation"):
        log_tags(stage, model_type, "test", {"tuning_run_id": tuning_run_id})
        log_metrics(results.metrics)
        log_plots(results, model_type)
"""