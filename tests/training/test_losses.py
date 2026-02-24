import tensorflow as tf
import numpy as np

from deep_chest.losses.weighted_bce import weighted_bce



def test_weighted_bce_returns_scalar():
    """The loss does reduce_mean, so output must be scalar """
    pos_weights = tf.constant([1.0, 2.0])
    loss_fn = weighted_bce(pos_weights)

    y_true = tf.constant([[1.0, 0.0]])
    logits = tf.constant([[0.5, -0.5]])

    loss = loss_fn(y_true, logits)

    assert loss.shape == ()




def test_weighted_bce_positive_weight_effect():
    """
    Higher pos_weight increases positive loss
    This verifies pos_weight actually influences loss
    """
    y_true = tf.constant([[1.0]])
    logits = tf.constant([[0.0]])  # neutral logit

    loss_small = weighted_bce(tf.constant([1.0]))(y_true, logits)
    loss_large = weighted_bce(tf.constant([5.0]))(y_true, logits)

    assert loss_large > loss_small



def test_weighted_bce_negative_class_not_scaled():
    """
    Zero positive label ignores pos_weight
    """
    y_true = tf.constant([[0.0]])
    logits = tf.constant([[0.0]])

    loss_small = weighted_bce(tf.constant([1.0]))(y_true, logits)
    loss_large = weighted_bce(tf.constant([5.0]))(y_true, logits)

    # should be equal for negative labels
    assert np.isclose(loss_small.numpy(), loss_large.numpy())



def test_weighted_bce_has_gradients():
    """
    Gradients exist
    """
    pos_weights = tf.constant([1.0])
    loss_fn = weighted_bce(pos_weights)

    logits = tf.Variable([[0.5]])
    y_true = tf.constant([[1.0]])

    with tf.GradientTape() as tape:
        loss = loss_fn(y_true, logits)

    grads = tape.gradient(loss, logits)

    assert grads is not None