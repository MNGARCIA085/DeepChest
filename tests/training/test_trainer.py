import tensorflow as tf
import pytest

from deep_chest.training.base import (
    StandardTrainer,
    TransferLearningTrainer,
    TrainingConfig,
    PhaseConfig,
)


"""
For trainers, we care about:

- compile is called correctly
- freeze/unfreeze logic works
- phase sequencing works
- returned structure is correct

We do NOT want heavy training.
We mock fit_model so tests run instantly.
"""



# ---- helper model ----
def build_dummy_model():
    # backbone
    bb_inputs = tf.keras.Input(shape=(10,))
    bb_x = tf.keras.layers.Dense(5)(bb_inputs)
    backbone = tf.keras.Model(
        bb_inputs,
        bb_x,
        name="backbone"
    )

    # outer model
    inputs = tf.keras.Input(shape=(10,))
    x = backbone(inputs)   # <-- important
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    return model


#------------Tests--------------#

def test_standard_trainer_returns_structure(monkeypatch):
    """ Standard trainer returns expected strcuture """
    model = build_dummy_model()
    cfg = TrainingConfig(
        phase1=PhaseConfig(lr=1e-3, epochs=1)
    )

    trainer = StandardTrainer(model, cfg)

    class DummyHistory:
        history = {"loss": [0.5]}

    monkeypatch.setattr(
        trainer,
        "fit_model",
        lambda *args, **kwargs: DummyHistory()
    )

    result = trainer.train(
        train_gen=None,
        val_gen=None,
        loss="mse",
        callbacks=[]
    )

    assert "history" in result
    assert result["hyperparams"]["lr"] == 1e-3




def test_transfer_trainer_requires_backbone():
    """ TransferLearningTrainer requires backbone_name """
    model = build_dummy_model()

    cfg = TrainingConfig(backbone_name=None)

    with pytest.raises(ValueError):
        TransferLearningTrainer(model, cfg)





def test_freeze_backbone():
    """ Freeze backbone actually freezes layers """
    model = build_dummy_model()
    cfg = TrainingConfig()

    trainer = TransferLearningTrainer(model, cfg)

    trainer.freeze_backbone()

    for layer in trainer.backbone.layers:
        assert layer.trainable is False




def test_unfreeze_top_layers():
    """ Unfreeze_top_layers works """
    model = build_dummy_model()
    cfg = TrainingConfig()

    trainer = TransferLearningTrainer(model, cfg)

    trainer.freeze_backbone()
    trainer.unfreeze_top_layers(1)

    # last layer should be trainable
    assert trainer.backbone.layers[-1].trainable is True



def test_transfer_trainer_two_phases(monkeypatch):
    """
    Transfer trainer executes both phases.
    We mock _phase1 and _phase2
    """
    model = build_dummy_model()
    cfg = TrainingConfig()

    trainer = TransferLearningTrainer(model, cfg)

    monkeypatch.setattr(
        trainer,
        "_phase1",
        lambda *args, **kwargs: (
            type("H", (), {"history": {"loss": [1.0]}})(),
            1e-4,
            2
        )
    )

    monkeypatch.setattr(
        trainer,
        "_phase2",
        lambda *args, **kwargs: (
            type("H", (), {"history": {"loss": [0.5]}})(),
            1e-5,
            1,
            1
        )
    )

    result = trainer.train(
        train_gen=None,
        val_gen=None,
        loss="mse",
        callbacks=[]
    )

    assert "phase1" in result
    assert "phase2" in result
    assert result["phase2"]["num_unfrozen"] == 1