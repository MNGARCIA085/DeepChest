import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB3,
)



@dataclass
class EfficientNetConfig:
    input_shape: tuple = (320, 320, 3)
    num_classes: int = 14
    train_backbone: bool = False
    dense_units: int = 256
    dropout: float = 0.5

    backbone_key: str = "efficientnet_b0"
    backbone_variant: str = "B0" # → architecture choice
    backbone_name: str = "backbone" # → layer name in the graph

    weights: str | None = "imagenet"



BACKBONE_REGISTRY = {
    "efficientnet_b0": EfficientNetB0,
    "efficientnet_b3": EfficientNetB3,
}



def build_efficientnet(cfg: EfficientNetConfig):

    backbone_cls = BACKBONE_REGISTRY[cfg.backbone_key]
    backbone_base = backbone_cls(
        include_top=False,
        weights=cfg.weights,
        input_shape=cfg.input_shape,
    )

    backbone_base.trainable = cfg.train_backbone

    # Wrap to give stable name
    backbone = tf.keras.Model(
        inputs=backbone_base.input,
        outputs=backbone_base.output,
        name=cfg.backbone_name
    )

    inputs = layers.Input(shape=cfg.input_shape)
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(cfg.num_classes)(x)

    return models.Model(inputs, outputs, name="classifier")


