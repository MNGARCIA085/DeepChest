import tensorflow as tf
from deep_chest.training.callbacks import early_stopping


def test_early_stopping_config():
    """ Verifies config propagation """
    cb = early_stopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=False
    )

    assert isinstance(cb, tf.keras.callbacks.EarlyStopping)
    assert cb.monitor == "val_accuracy"
    assert cb.patience == 3
    assert cb.restore_best_weights is False