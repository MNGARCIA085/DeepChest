import numpy as np
import pytest
import tensorflow as tf

from deep_chest.preprocessors.base import DataModule
from deep_chest.inference.base import Predictor
from deep_chest.inference.pipeline import InferencePipeline
from deep_chest.inference.builder import build_pipeline_from_components



"""
Notes:
    - Generator test: predict_with_targets expects a generator, so we mock a tiny one.
    - Shape checks: Always verify (batch, num_classes) and probabilities are in [0,1].
    - Single vs batch inputs: Covers predict_one_probs and batch predictions.
    - Pipeline tests: Ensure InferencePipeline correctly preprocesses and calls the predictor.
"""





# --------- Fixtures ---------
@pytest.fixture
def dummy_model():
    # tiny multi-label binary model
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(3)(x)  # 3 labels
    model = tf.keras.Model(inputs, outputs)
    return model

@pytest.fixture
def dummy_data_module():
    class DummyPrep:
        def preprocess(self, x):
            # accept single array or batch
            if isinstance(x, np.ndarray):
                if x.ndim == 3:
                    x = np.expand_dims(x, 0)  # add batch
            return x
    return DummyPrep()


@pytest.fixture
def predictor(dummy_model):
    return Predictor(dummy_model)


@pytest.fixture
def pipeline(dummy_data_module, dummy_model):
    predictor = Predictor(dummy_model)
    return InferencePipeline(dummy_data_module, predictor)


# --------- Tests Predictor ---------
def test_predict_logits_shape(predictor):
    x = np.random.rand(2, 32, 32, 3).astype(np.float32)
    logits = predictor.predict_logits(x)
    assert logits.shape == (2, 3)

def test_predict_probs_range(predictor):
    x = np.random.rand(2, 32, 32, 3).astype(np.float32)
    probs = predictor.predict_probs(x)
    assert probs.shape == (2, 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)

def test_predict_one_logits_shape(predictor):
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    logits = predictor.predict_one_logits(x)
    assert logits.shape == (1, 3)

def test_predict_one_probs_range(predictor):
    x = np.random.rand(1, 32, 32, 3).astype(np.float32)
    probs = predictor.predict_one_probs(x)
    assert probs.shape == (1, 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# --------- Tests predict_with_targets ----------
def test_predict_with_targets_batch(predictor):
    # tiny generator: yields (X, y)
    X = np.random.rand(4, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 2, (4, 3))
    def gen():
        for i in range(4):
            yield X[i:i+1], y[i:i+1]

    probs, targets = predictor.predict_with_targets(gen(), steps=4)
    assert probs.shape == (4, 3)
    assert targets.shape == (4, 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# --------- Tests InferencePipeline ----------
def test_pipeline_predict_shape(pipeline):
    x = np.random.rand(32, 32, 3).astype(np.float32)
    probs = pipeline.predict(x)
    assert probs.shape == (1, 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_pipeline_predict_batch(pipeline):
    x = np.random.rand(2, 32, 32, 3).astype(np.float32)
    probs = pipeline.predict(x)
    assert probs.shape == (2, 3)
    assert np.all(probs >= 0) and np.all(probs <= 1)

