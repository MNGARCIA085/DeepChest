from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model



@dataclass
class DenseNetConfig:
    input_shape: tuple = (320, 320, 3)
    num_classes: int = 14
    train_backbone: bool = False
    backbone_name: str = "backbone" # → layer name in the graph
    weights: str | None = "imagenet"



def build_densenet(cfg: DenseNetConfig):
    # backbone base
    backbone_base = DenseNet121(
        include_top=False,
        weights=cfg.weights,
        #input_shape=cfg.input_shape,
    )

    backbone_base.trainable = False

    # Wrap backbone with stable name
    backbone = tf.keras.Model(
        inputs=backbone_base.input,
        outputs=backbone_base.output,
        name=cfg.backbone_name
    )

    inputs = tf.keras.layers.Input(shape=cfg.input_shape)
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(cfg.num_classes, activation=None)(x)

    model = tf.keras.Model(inputs, outputs, name="classifier")

    return model


# note -> bc of the CPU i need to train with a small batch size (e.g. 4)
# ill use for ex bs=4 when i prepare the datasets
