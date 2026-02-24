import pytest
import numpy as np
import tensorflow as tf
from deep_chest.preprocessors.base import DataModule


def test_preprocess_output_shape():
    """
    Confirms
        - resizing works
        - batch dimension added
        - dtype corr
    """
    dm = DataModule(image_size=(320, 320))

    dummy_img = np.random.randint(
        0, 255, size=(500, 400, 3), dtype=np.uint8
    )

    x = dm.preprocess(dummy_img)

    assert x.shape == (1, 320, 320, 3)
    assert x.dtype == tf.float32



def test_preprocess_removes_alpha_channel():
    """ If someone uploads a PNG with alpha channel, you’re protected. """
    dm = DataModule(image_size=(224, 224))

    dummy_rgba = np.random.randint(
        0, 255, size=(300, 300, 4), dtype=np.uint8
    )

    x = dm.preprocess(dummy_rgba)

    assert x.shape == (1, 224, 224, 3)


#-----compute pos_weights works-----------#
class DummyGen:
    def __init__(self, labels):
        self.labels = labels


def test_compute_pos_weights():
    """
    This validates:
        - shape correctness
        - frequencies bounded
        - no crash on division
    """
    labels = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 0],
    ])

    gen = DummyGen(labels)

    pw, pos_f, neg_f = DataModule.compute_pos_weights_from_generator(gen)

    assert len(pw) == 2
    assert np.all(pos_f <= 1)
    assert np.all(pos_f >= 0)





def test_train_generator_without_loading():
    """error when generator used before loading"""
    dm = DataModule()

    with pytest.raises(RuntimeError):
        dm.train_generator()