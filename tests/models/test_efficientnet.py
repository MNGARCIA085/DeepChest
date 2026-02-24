import tensorflow as tf
from deep_chest.models.efficient_net import (
    build_efficientnet,
    EfficientNetConfig,
)


"""
I will test:
    - Transfer learning freezing logic works
    - Config truly drives architecture
    - Registry pattern works
    - Forward pass stable
    - Model scalable across batch sizes
"""



def test_efficientnet_output_shape():
    """Model builds and output shape correct"""
    cfg = EfficientNetConfig(
        input_shape=(320, 320, 3),
        num_classes=14,
        weights=None, # avoid downloading
        backbone_key="efficientnet_b0",
    )

    model = build_efficientnet(cfg)

    x = tf.random.normal((1, 320, 320, 3))
    y = model(x)

    assert y.shape == (1, 14)



#------Backbone freezing works---------#
def test_backbone_layers_frozen():
    cfg = EfficientNetConfig(
        train_backbone=False,
        weights=None
    )
    model = build_efficientnet(cfg)

    backbone = model.get_layer(cfg.backbone_name)

    assert all(not layer.trainable for layer in backbone.layers)


def test_backbone_layers_trainable():
    cfg = EfficientNetConfig(
        train_backbone=True,
        weights=None
    )
    model = build_efficientnet(cfg)

    backbone = model.get_layer(cfg.backbone_name)

    assert any(layer.trainable for layer in backbone.layers)


#-------------------------------------#




def test_backbone_registry_selection():
    """This confirms your registry logic works."""
    cfg = EfficientNetConfig(backbone_key="efficientnet_b3")
    model = build_efficientnet(cfg)

    backbone = model.get_layer(cfg.backbone_name)

    # EfficientNetB3 has larger output channels than B0
    output_channels = backbone.output_shape[-1]

    assert output_channels >= 1536  # B3 is larger than B0 (1280)



def test_dense_units_respected():
    """Dense head responds to config"""
    cfg = EfficientNetConfig(dense_units=512)
    model = build_efficientnet(cfg)

    dense_layers = [
        l for l in model.layers
        if isinstance(l, tf.keras.layers.Dense)
    ]

    # First Dense after BN should have 512 units
    assert dense_layers[0].units == 512


def test_forward_pass_batch_size():
    """Batch flexibility"""
    cfg = EfficientNetConfig()
    model = build_efficientnet(cfg)

    x = tf.random.normal((4, 320, 320, 3))
    y = model(x)

    assert y.shape[0] == 4