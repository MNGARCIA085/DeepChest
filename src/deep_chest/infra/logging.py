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

        # RUN_ID
        run_id = mlflow.active_run().info.run_id

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


        print(results)

        #----------------------------------------------------#

        # for now keep it simple; last auprc
        from deep_chest.infra.utils import update_leaderboard


        # is differnet for TL!!!!!! (if history in results.....)
        if 'history' in results:
            score = results['history']['val_auprc'][-1]
        else: #TL
            score = results['phase2']['history']['val_auprc'][-1]
        
        # update leaderboard
        info = update_leaderboard(run_id, score)

        if info["is_top_k"]:
            log_keras_model(model)
            mlflow.set_tag("top_k_rank", info["rank"])
            mlflow.set_tag("in_top_k", "true")
        else:
            mlflow.set_tag("in_top_k", "false")

        
        #for rid in info["removed_run_ids"]:
        #    delete_model_artifacts(rid)
        #mlflow.log_artifact("mlflow_leaderboard.json") correct path



        



from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import mlflow

def log_keras_model(model, filename="model.keras"):
    # Hydra unique output dir
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = out_dir / filename

    # 1. Save locally (.keras format)
    model.save(model_path) #keras_model_kwargs={"include_optimizer": False}

    # 2. Log to MLflow under "model" folder
    mlflow.log_artifact(str(model_path), artifact_path="model")

    return str(model_path)




        



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