import tensorflow as tf
from tensorflow.keras import layers, models


from dataclasses import dataclass



@dataclass
class CNNConfig:
    input_shape: tuple = (320, 320, 3)
    num_classes: int = 14

    filters: tuple = (32, 64, 128)
    dense_units: int = 128

    dropout: float = 0.5
    activation: str = "relu"

    kernel_initializer: str = "he_normal"
    weight_decay: float = 1e-4 # more of a training thing

    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5

    name: str = 'simple_cnn_baseline'





from tensorflow.keras import layers, models, regularizers

def build_cnn(cfg: CNNConfig):

    reg = regularizers.l2(cfg.weight_decay)

    inputs = layers.Input(shape=cfg.input_shape)
    x = inputs

    for f in cfg.filters:
        x = layers.Conv2D(
            f, 3,
            padding="same",
            use_bias=False,
            kernel_initializer=cfg.kernel_initializer,
            kernel_regularizer=reg
        )(x)

        x = layers.BatchNormalization(
            momentum=cfg.bn_momentum,
            epsilon=cfg.bn_epsilon
        )(x)

        x = layers.Activation(cfg.activation)(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        cfg.dense_units,
        activation=cfg.activation,
        kernel_initializer=cfg.kernel_initializer,
        kernel_regularizer=reg
    )(x)

    x = layers.Dropout(cfg.dropout)(x)

    outputs = layers.Dense(
        cfg.num_classes,
        kernel_initializer=cfg.kernel_initializer
    )(x)

    return models.Model(inputs, outputs, name=cfg.name)
