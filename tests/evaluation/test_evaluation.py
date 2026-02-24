import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score
from deep_chest.evaluation.evaluator import Evaluator

# dummy data
labels = ["A", "B", "C"]
thresholds = [0.5, 0.5, 0.5]

# small batch of 4 examples
y_true = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
])

y_pred_proba = np.array([
    [0.9, 0.2, 0.7],
    [0.1, 0.8, 0.3],
    [0.8, 0.7, 0.2],
    [0.2, 0.1, 0.9],
])


@pytest.fixture
def evaluator():
    return Evaluator(threshold=thresholds, labels=labels)


def test_accuracy_per_class(evaluator):
    y_hat = (y_pred_proba >= thresholds).astype(int)
    acc = evaluator.accuracy_per_class(y_true, y_hat)
    assert acc.shape[0] == len(labels)
    assert np.all(acc >= 0) and np.all(acc <= 1)


def test_recall_per_class(evaluator):
    y_hat = (y_pred_proba >= thresholds).astype(int)
    rec = evaluator.recall_per_class(y_true, y_hat)
    assert rec.shape[0] == len(labels)
    assert np.all(rec >= 0) and np.all(rec <= 1)


def test_specificity_ppv_npv(evaluator):
    y_hat = (y_pred_proba >= thresholds).astype(int)
    spec = evaluator.specificity_per_class(y_true, y_hat)
    ppv = evaluator.ppv_per_class(y_true, y_hat)
    npv = evaluator.npv_per_class(y_true, y_hat)
    for arr in [spec, ppv, npv]:
        assert arr.shape[0] == len(labels)
        assert np.all(arr >= 0) and np.all(arr <= 1)


def test_auroc_auprc(evaluator):
    auroc = evaluator.auroc_per_class(y_true, y_pred_proba)
    auprc = evaluator.auprc_per_class(y_true, y_pred_proba)
    for arr in [auroc, auprc]:
        assert len(arr) == len(labels)
        assert np.all(np.array(arr) >= 0) and np.all(np.array(arr) <= 1)


def test_f1_per_class(evaluator):
    y_hat = (y_pred_proba >= thresholds).astype(int)
    f1 = evaluator.f1_per_class(y_true, y_hat)
    assert f1.shape[0] == len(labels)
    assert np.all(f1 >= 0) and np.all(f1 <= 1)


def test_evaluate_returns_dicts(evaluator):
    per_class, agg = evaluator.evaluate(y_true, y_pred_proba)
    assert isinstance(per_class, dict)
    assert isinstance(agg, dict)
    for key in ["accuracy", "recall", "sepcificity", "f1", "prevalence", "ppv_per_class", "npv_per_class", "auroc", "auprc"]:
        assert key in per_class
    for key in ["accuracy_macro", "recall_macro", "auroc_micro"]:
        assert key in agg


def test_bootstrap_auc(evaluator):
    df = evaluator.bootstrap_auc(y_true, y_pred_proba, bootstraps=5, fold_size=4)
    assert isinstance(df, (type(None), object)) or hasattr(df, "iloc")
    assert df.shape[0] == len(labels)