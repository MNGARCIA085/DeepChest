import mlflow
import joblib
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import is_dataclass, asdict
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
    mlflow.log_param("image_size", artifacts["image_size"])
    mlflow.log_param("num_labels", artifacts["num_labels"])
    mlflow.log_param("preprocessing", artifacts["preprocessing"])
    mlflow.log_param("seed", artifacts["seed"])

    # labels artifact
    mlflow.log_dict(
        {"labels": artifacts["labels"]},
        "labels.json"
    )

    # pos weigths
    pos_weights_np = np.array(artifacts['pos_weights'])
    mlflow.log_dict(
        {"pos_weights": pos_weights_np.tolist()},
        "pos_weights.json"
    )


# hyperparams
def log_hyperparams(hyperparams: dict, prefix=""):
    for k, v in hyperparams.items():
        mlflow.log_param(f"{prefix}{k}", v)





#---------------Training results-------------------#
def log_loss_plot(history, filename="loss.png"):
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





def log_training_results(results):

    # -------- STANDARD --------
    if "history" in results:
        log_hyperparams(results.get("hyperparams", {}))
        log_loss_plot(results["history"])
        return

    # -------- TRANSFER LEARNING --------
    for phase, data in results.items():

        if phase == "metrics":
            continue

        if not isinstance(data, dict):
            continue

        # log params
        phase_params = {k: v for k, v in data.items() if k != "history"}
        log_hyperparams(phase_params, prefix=f"{phase}_")

        # log loss
        if "history" in data:
            log_loss_plot(data["history"], filename=f"{phase}_loss.png")


#---------------------Metrics-----------------------------#
def log_per_class_metrics(metrics: dict, prefix=""): # prefix=metrics
    for metric_name, arr in metrics.items():
        for i, v in enumerate(arr):
            mlflow.log_metric(
                f"{prefix}_{metric_name}_class_{i}",
                float(v)
            )


def build_per_class_df(metrics: dict):
    df = pd.DataFrame(metrics)
    df.index.name = "class_id"
    return df


def log_per_class_df(df, filename="per_class_metrics.csv"):
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    df.to_csv(path, index=True)
    mlflow.log_artifact(path)


def log_metric_means(metrics, prefix=""):
    for name, arr in metrics.items():
        mlflow.log_metric(
            f"{prefix}_{name}_mean",
            float(np.mean(arr))
        )



#--------------------Model------------------------------------#
def log_model_config(config, prefix="model_"):
    if not is_dataclass(config):
        raise ValueError("config must be dataclass")

    cfg_dict = asdict(config)

    for k, v in cfg_dict.items():
        # tuples/lists → string
        if isinstance(v, (tuple, list)):
            v = str(v)

        mlflow.log_param(f"{prefix}{k}", v)

"""
def log_model_config_artifact(config):
    cfg_dict = asdict(config)
    mlflow.log_dict(cfg_dict, "model_config.json")
"""



def log_model_summary(model, filename="model_summary.txt"):
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    with open(path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    mlflow.log_artifact(path)




def log_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mlflow.log_metric("params_total", total)
    mlflow.log_metric("params_trainable", trainable)




#-------------------Combined plot for TL------------------------#
def log_transfer_loss_plot(results, filename="combined_loss.png"):
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    p1 = results["phase1"]["history"]
    p2 = results["phase2"]["history"]

    train1, val1 = p1["loss"], p1.get("val_loss", [])
    train2, val2 = p2["loss"], p2.get("val_loss", [])

    # concatenate
    train_all = train1 + train2
    val_all = val1 + val2

    split_epoch = len(train1)

    plt.figure()

    plt.plot(train_all, label="train_loss")
    if val_all:
        plt.plot(val_all, label="val_loss")

    # vertical line
    plt.axvline(
        x=split_epoch - 0.5,
        linestyle="--",
        label="fine_tuning_start"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Feature Extraction → Fine Tuning")
    plt.legend()

    plt.savefig(path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(path)





#-------------------Main code-------------------------------#
def logging(run_name, artifacts, results, model, model_cfg, model_type, stage):
    with mlflow.start_run(run_name=run_name):
        
        # tags
        log_tags(stage, model_type, "train/val")

        # prep. artifacts
        log_common_params(artifacts)

        # model
        log_model_config(model_cfg)
        log_model_summary(model)
        #log_param_count(model)

        # training res.
        log_training_results(results)

        # TL plot
        if "phase1" in results and "phase2" in results:
            log_transfer_loss_plot(results)

        # df with metrics per class
        df = build_per_class_df(results['metrics'])
        log_per_class_df(df) # later put class name instead of class_id

        # metrics aggregation
        log_metric_means(results['metrics'])

        


        





"""




{'phase1': {'history': {'auprc': [0.05389304831624031], 'auroc': [0.5418590903282166], 'loss': [1.6506065130233765], 'val_auprc': [0.0867825299501419], 'val_auroc': [0.6071411967277527], 'val_loss': [1.3783063888549805]}, 'lr': 0.0001, 'epochs': 1}, 'phase2': {'history': {'auprc': [0.056598179042339325], 'auroc': [0.5636993646621704], 'loss': [1.602590560913086], 'val_auprc': [0.08525221049785614], 'val_auroc': [0.651050865650177], 'val_loss': [1.3812135457992554]}, 'lr': 1e-05, 'epochs': 1, 'num_unfrozen': 2}, 'metrics': {'accuracy': array([0.54 , 0.82 , 0.735, 0.925, 0.53 , 0.3  , 0.35 , 0.33 , 0.745,
       0.605, 0.71 , 0.835, 0.63 , 0.485]), 'recall': array([0.25      , 0.66666667, 0.51851852, 0.        , 0.62068966,
       0.88888889, 0.63636364, 0.94736842, 0.5       , 0.66666667,
       1.        , 1.        , 0.5       , 0.71428571]), 'sepcificity': array([0.54591837, 0.82233503, 0.76878613, 0.92964824, 0.51461988,
       0.27225131, 0.33333333, 0.26519337, 0.75789474, 0.60309278,
       0.70854271, 0.83417085, 0.63131313, 0.47668394]), 'f1': array([0.0212766 , 0.1       , 0.34567901, 0.        , 0.27692308,
       0.1025641 , 0.09722222, 0.21176471, 0.16393443, 0.09195402,
       0.03333333, 0.05714286, 0.02631579, 0.08849558]), 'prevalence': array([0.02 , 0.015, 0.135, 0.005, 0.145, 0.045, 0.055, 0.095, 0.05 ,
       0.03 , 0.005, 0.005, 0.01 , 0.035]), 'ppv_per_class': array([0.01111111, 0.05405405, 0.25925926, 0.        , 0.17821782,
       0.05442177, 0.05263158, 0.1192053 , 0.09803922, 0.04938272,
       0.01694915, 0.02941176, 0.01351351, 0.04716981]), 'npv_per_class': array([0.97272727, 0.99386503, 0.9109589 , 0.99462366, 0.88888889,
       0.98113208, 0.94029851, 0.97959184, 0.96644295, 0.98319328,
       1.        , 1.        , 0.99206349, 0.9787234 ]), 'auc': array([0.45153061, 0.72081218, 0.71397988, 0.32663317, 0.60092761,
       0.65212333, 0.4968735 , 0.67199767, 0.65736842, 0.65034364,
       0.94974874, 0.97487437, 0.7020202 , 0.63582531])}}
"""





        

"""
def log_history(history: dict, prefix=""):
    for metric_name, values in history.items():
        for epoch, v in enumerate(values):
            mlflow.log_metric(f"{prefix}{metric_name}", v, step=epoch)
"""




"""
#------------------Final evaluation (with test set)-----------------------------
def log_test_results(tuning_run_id, model_type, results, stage='eval'):
    with mlflow.start_run(run_name="test_evaluation"):
        log_tags(stage, model_type, "test", {"tuning_run_id": tuning_run_id})
        log_metrics(results.metrics)
        log_plots(results, model_type)
"""