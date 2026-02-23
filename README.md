# Chest X-Ray Multi-Label Classification

## Project Overview

This project builds a deep learning system to predict 14 thoracic diseases from chest X-ray images. The task is formulated as a **multi-label classification problem**, where each image may contain multiple pathologies simultaneously.

The goal is not to reach state-of-the-art performance, but to demonstrate solid ML engineering practices in a realistic, high-dimensional, imbalanced setting.

Key aspects of the project:

- Multi-label image classification (14 binary outputs)
- Class imbalance awareness
- Mean validation AUPRC (`val_auprc_mean`) as primary metric
- Leaderboard-based model selection
- GPU training via Google Colab
- Reproducible experiments using Hydra
- Lightweight deployment with a Gradio web app

The focus is on clarity, reproducibility, and disciplined experiment management rather than maximizing leaderboard scores.



## Metric & Model Selection Strategy

Due to strong class imbalance, the primary validation metric is:

- **Mean Area Under the Precision-Recall Curve (AUPRC)** across the 14 labels

Accuracy and ROC-AUC are intentionally not used as primary metrics because they can be misleading in imbalanced multi-label settings.

A lightweight **leaderboard file** is used to:

- Track trained models
- Rank them by `val_auprc_mean`
- Retain only the top-k models

Since image models are large, inferior models are not stored to avoid unnecessary artifact accumulation. This mimics basic model registry behavior without introducing unnecessary infrastructure for a portfolio project.



## Project Structure

```
project-root/
│
├── scripts/ # Training, evaluation, inference
├── conf/ # Hydra configuration files
├── src/ # Core model and data logic
├── notebooks/ # EDA, Colab training notebook
├── tracking/ # Saved top-k models
├── requirements.txt
├── gradio_app/ # Simple gradio app
```

## Setup

Create virtual environment and install dependencies

```
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -e .
```

## Training

Training is executed via Hydra:

```
python -m scripts.training model_type=cnn
```

You can override parameters directly from the CLI:

```
python -m scripts.training model_type=cnn model_type.training.phase1.epochs=10
python -m scripts.training model_type=efficientnet model_type.training.phase2.lr=1e-5
```

After training:
- Validation AUPRC is computed
- The model is compared against the leaderboard
- Only top-k models are retained in artifacts


## GPU Training (Colab)

Because image models require GPU acceleration, training can be performed in Google Colab.

The notebook "train_colab.ipynb":

- Clones the repository

- Installs minimal dependencies

- Launches training via: ```python -m scripts.training model_type=cnn tracking.enabled=false```

If a trained model enters the top-k leaderboard, it can be saved locally and can be downloaded for later promotion.

Colab is used purely as a compute backend. All core logic remains inside the repository.


## Experiment tracking

Local experiment tracking is performed using MLflow.

Example:

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host localhost \
  --port 5000


For simplicity, a remote MLflow server is not deployed in this portfolio project.
Models trained in Colab can later be promoted and logged locally if desired.


## Web demo

A lightweight web interface is built using Gradio to demonstrate real-time inference.

The app allows users to upload a chest X-ray image and receive predicted probabilities for the 14 conditions.



## Included Demo Artifacts

The repository includes:

- A lightweight sample model checkpoint
- Associated metadata
- A sample chest X-ray image

These artifacts are provided to facilitate environment emulation and local testing without requiring full training.

For example, the Gradio app loads the model directly from disk, allowing the inference pipeline and UI to be tested immediately after setup. This keeps the repository self-contained and reproducible while avoiding the need to download large training artifacts.

## Design Philosophy

This project emphasizes:
- Reproducibility (Hydra configs)
- Metric awareness (AUPRC for imbalance)
- Clean training pipeline
- Controlled artifact retention
- Clear separation of training and inference
- Practical deployment demonstration

The objective is to show applied ML engineering skills in a realistic medical imaging scenario, not to compete for benchmark state-of-the-art results.



