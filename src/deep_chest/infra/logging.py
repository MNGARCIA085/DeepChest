import mlflow
import joblib
import os
import json

from dataclasses import is_dataclass, asdict
#from net_security.utils.plots import plot_cm, plot_roc, plot_train_val

from hydra.core.hydra_config import HydraConfig


"""
from hydra.core.hydra_config import HydraConfig
import os

out_dir = HydraConfig.get().runtime.output_dir
plot_path = os.path.join(out_dir, "roc.png")

plt.savefig(plot_path)
mlflow.log_artifact(plot_path)
"""



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
    mlflow.log_param("image_size", artifacts["image_size"])
    mlflow.log_param("num_labels", artifacts["num_labels"])
    mlflow.log_param("preprocessing", artifacts["preprocessing"])
    mlflow.log_param("seed", artifacts["seed"])

    # labels artifact
    mlflow.log_dict(
        {"labels": artifacts["labels"]},
        "labels.json"
    )

    # labels
    """
    out_dir = HydraConfig.get().runtime.output_dir
    filename = "labels.json"
    path = os.path.join(out_dir, filename)
    mlflow.log_dict({"labels": artifacts["labels"]}, path)
    """

    # pos weigths
    pos_weights_np = np.array(artifacts['pos_weights'])

    mlflow.log_dict(
        {"pos_weights": pos_weights_np.tolist()},
        "pos_weights.json"
    )




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



        log_loss_plot(results['history'])

    else:  # Transfer learning

        for phase, data in results.items():
            log_hyperparams(
                {k: v for k, v in data.items() if k != "history"},
                prefix=f"{phase}_"
            )
            #log_final_metrics(data["history"], prefix=f"{phase}_")
            #log_history(data["history"], prefix=f"{phase}_")

            print('sdfdsfdsdsfdsdsf')
            print(phase)
            print(data)
            print(type(phase))
            print(type(data))

            print(data.items())
            print(type(data.items()))


            # histories
            for subphase, subdata in data.items():
                if "history" in subdata:

                    log_loss_plot(subdata['history'])
                    """
                    log_final_metrics(
                        subdata["history"],
                        prefix=f"{phase}_{subphase}_"
                    )
                    """

            print(phase)

            print("-------------------------------\n--------------------")
            print(data)
            log_loss_plot(data['history'])

            # phase1['history']



"""
better for hydra
for phase, phase_data in results.items():
    for key, value in phase_data.items():
        if isinstance(value, dict) and "history" in value:
            log_final_metrics(value["history"], prefix=f"{phase}_{key}_")
"""



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
import os
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig



def log_loss_plot(history, filename="loss.png"):


    print(history)

    # 1. Get hydra output dir (unique per run)
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    # 2. Extract history safely
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    # 3. Plot
    plt.figure()
    plt.plot(train_loss, label="train_loss")
    if val_loss:
        plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    # 4. Save locally (hydra safe)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    # 5. Log to MLflow
    mlflow.log_artifact(path)





#------------------------------------------------#






# log exp. log_Experiment
def logging(run_name, artifacts, results, model_type, stage):

    with mlflow.start_run(run_name=run_name):
        log_tags(stage, model_type, "train/val")
        """
        
        log_metrics(results)
        log_hyperparams(results['hyperparams'])
        """
        #log_per_class_metrics(results['metrics'])


        # check mutability

        log_training_results(results)


        # prep. artifacts
        log_common_params(artifacts)

        df = build_per_class_df(results['metrics'])
        log_per_class_df(df) # later pus class name instead of class_id

        # log metrics aggregate
        log_metric_means(results['metrics'])

        











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