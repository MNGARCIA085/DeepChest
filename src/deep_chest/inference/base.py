import numpy as np
import tensorflow as tf

# some models might havbe batch norm, have it in count; nothing to do, training=False is enough


class Predictor:
    """
    Predictor for multi-label binary models (sigmoid outputs).
    Thin wrapper around tf.keras.Model.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    # ---------- Batch preds ----------
    def predict_logits(self, generator, steps=None):
        return self.model.predict(generator, steps=steps)

    def predict_probs(self, generator, steps=None):
        logits = self.predict_logits(generator, steps=steps)
        return tf.nn.sigmoid(logits).numpy()


    def predict_with_targets(self, generator, steps=None):
        if steps is None:
            if hasattr(generator, "__len__"):
                steps = len(generator)
            else:
                raise ValueError("steps must be set for infinite generators")

        probs, targets = [], []

        for i, (x, y) in enumerate(generator):
            if i >= steps:
                break

            logits = self.model(x, training=False)
            probs.append(tf.sigmoid(logits).numpy())
            targets.append(y)

        return np.vstack(probs), np.vstack(targets)


    # ---------- single sample ----------
    def predict_one_logits(self, x):
        """
        x: np.ndarray or tf.Tensor of shape (1, H, W, C)
        """
        logits = self.model(x, training=False)
        return logits.numpy()

    def predict_one_probs(self, x):
        logits = self.predict_one_logits(x)
        return tf.nn.sigmoid(logits).numpy()


"""
better version
def predict_with_targets(self, generator, steps=None):
if steps is None:
    steps = len(generator) if hasattr(generator, "__len__") else None
    if steps is None:
        raise ValueError("steps must be set for infinite generators")

all_probs = []
all_targets = []

# Iterate once
for i, (x, y) in enumerate(generator):
    if i >= steps:
        break
    
    # Keep predictions on device as long as possible
    logits = self.model(x, training=False)
    probs = tf.sigmoid(logits)
    
    all_probs.append(probs.numpy())
    all_targets.append(y)

return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)
"""