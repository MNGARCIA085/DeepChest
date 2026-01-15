import tensorflow as tf


def early_stopping(
    monitor: str = "val_loss",
    patience: int = 5,
    restore_best_weights: bool = True,
):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
    )


def model_checkpoint(
    filepath: str,
    monitor: str = "val_loss",
    save_best_only: bool = True,
):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
    )


def lr_logging():
    return LrLoggingCallback()
