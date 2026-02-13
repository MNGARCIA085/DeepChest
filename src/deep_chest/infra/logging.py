import mlflow
import joblib
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
from dataclasses import is_dataclass, asdict
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from mlflow.tracking import MlflowClient
from urllib.parse import urlparse
from deep_chest.infra.utils import update_leaderboard, load_leaderboard
from deep_chest.visualization.plots import build_transfer_loss_plot, build_precision_recall_plot, build_loss_plot



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
def log_training_results(results):

    # -------- STANDARD --------
    if "history" in results:
        log_hyperparams(results.get("hyperparams", {}))
        fig = build_loss_plot(results['history'])
        log_figure(fig, "loss.png")
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
            fig = build_loss_plot(data['history'])
            log_figure(fig, f"{phase}_loss.png")




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
    print(f"Writing to unique dir: {out_dir}")  # Should be different for each parallel job
    path = os.path.join(out_dir, filename)

    df.to_csv(path, index=True)
    mlflow.log_artifact(path)


def log_aggregate_metrics(agg_metrics):
    for k, v in agg_metrics.items():
        mlflow.log_metric(k, float(v))



#------------------Confidence Interval-----------------------#
def log_ci(df, filename="ci.csv"): # maybe json later
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    df.to_csv(path, index=True)
    mlflow.log_artifact(path)



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
def log_figure(fig, filename):
    out_dir = HydraConfig.get().runtime.output_dir
    path = os.path.join(out_dir, filename)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(path)



#-------------Log Keras model-------------------#
def log_keras_model(model, filename="model.keras"):
    """
        I can't use directly mlflow's log_model in this case
    """
    # Hydra unique output dir
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = out_dir / filename

    # 1. Save locally (.keras format)
    model.save(model_path) #keras_model_kwargs={"include_optimizer": False}

    # 2. Log to MLflow under "model" folder
    mlflow.log_artifact(str(model_path), artifact_path="model")

    return str(model_path)




#-----------Log leaderboard-----------------------#
def log_leaderboard(leaderboard):
    mlflow.log_dict(
        {"leaderboard": leaderboard},
        "leaderboard.json"
    )





#--------------------Delete worst model so far-------------------#
def delete_model_artifact(run_id):
    client = MlflowClient()
    run = client.get_run(run_id)

    uri = run.info.artifact_uri
    parsed = urlparse(uri)
    base_path = Path(parsed.path)

    model_dir = base_path / "model"

    if model_dir.exists():
        shutil.rmtree(model_dir)





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
            fig = build_transfer_loss_plot(results)
            log_figure(fig, "combined_loss.png")

        # df with metrics per class
        df = build_per_class_df(results['metrics'])
        log_per_class_df(df) # later put class name instead of class_id

        # metrics aggregation
        log_aggregate_metrics(results['agg_metrics'])


        #----------------------------------------------------#
        # for now keep it simple; last auprc; later use checkpoint appropiately

        # is differnet for TL
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


        # log leaderboard (at that time, maybe later update appropiately)
        leaderboard = load_leaderboard()
        log_leaderboard(leaderboard)


        print(info["removed_run_ids"])
        # remove worst models
        for rid in info["removed_run_ids"]:
            delete_model_artifact(rid)
        




#--------------------Log test results--------------------------#
def log_test_results(model_type, per_class_metrics, agg_metrics, curves, ci, run_id, stage='test'):

    with mlflow.start_run(run_name="test_evaluation"):
        log_tags(stage, model_type, "test", {"run_id": run_id})
        df = build_per_class_df(per_class_metrics) # later put class name instead of class_id
        log_per_class_df(df) 

        # metrics aggregation
        log_aggregate_metrics(agg_metrics)

        # PR curve
        fig = build_precision_recall_plot(curves)
        log_figure(fig, "precision_recall_curve.png")

        # CI
        log_ci(ci)
