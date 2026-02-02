from dataclasses import dataclass



@dataclass
class EfficientNetConfig:
    input_shape: tuple = (320, 320, 3)
    num_classes: int = 14
    train_backbone: bool = False
    dense_units: int = 256
    dropout: float = 0.5

    backbone_key: str = "efficientnet_b0"
    backbone_variant: str = "B0"
    backbone_name: str = "backbone"



"""
backbone_name → layer name in the graph
backbone_variant → architecture choice
"""



from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB3,
    #ResNet50,
    #MobileNetV2,
)

BACKBONE_REGISTRY = {
    "efficientnet_b0": EfficientNetB0,
    "efficientnet_b3": EfficientNetB3,
    #"resnet50": ResNet50,
    #"mobilenetv2": MobileNetV2,
}



import tensorflow as tf
from tensorflow.keras import layers, models




def build_efficientnet(cfg: EfficientNetConfig):

    backbone_cls = BACKBONE_REGISTRY[cfg.backbone_key]
    backbone_base = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=cfg.input_shape,
    )

    backbone_base.trainable = cfg.train_backbone

    # Wrap to give stable name // if i put it above it doesnt dowload the model!
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



#model.get_layer("backbone")



"""
def build_efficientnet(cfg: EfficientNetConfig):

    
    backbone_cls = BACKBONE_REGISTRY[cfg.backbone_key]
    backbone = backbone_cls(
        include_top=False,
        weights="imagenet",
        #weights="/path/to/EfficientNetB0_notop.h5" -> download weights firts!!!
        #weights=None,
        input_shape=cfg.input_shape,
    )
    
    
    backbone.trainable = cfg.train_backbone

    inputs = layers.Input(shape=cfg.input_shape)
    #x = preprocess_input(inputs); see later; in or out, in in certain way is safer for inference
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(cfg.dense_units, activation="relu")(x)
    x = layers.Dropout(cfg.dropout)(x) # logits
    outputs = layers.Dense(cfg.num_classes)(x)

    return models.Model(inputs, outputs, name="efficientnet")
"""

# note -> it doesnt have a sigmoid here
# training=False when frozen


