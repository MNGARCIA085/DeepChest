import tensorflow as tf
from deep_chest.models.cnn import build_cnn, CNNConfig


def test_model_output_shape():
    """Model builds and output shape is correct"""
    cfg = CNNConfig(input_shape=(320, 320, 3), num_classes=14)
    model = build_cnn(cfg)

    x = tf.random.normal((1, 320, 320, 3))
    y = model(x)

    assert y.shape == (1, 14)



def test_model_name():
    """Model name is applied"""
    cfg = CNNConfig(name="test_model")
    model = build_cnn(cfg)

    assert model.name == "test_model"


def test_number_of_conv_layers():
    """Nimber of conv layers matches config"""
    cfg = CNNConfig(filters=(16, 32, 64, 128))
    model = build_cnn(cfg)

    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    assert len(conv_layers) == 4



def test_weight_decay_applied():
    """Weight decay actually applied (You don’t test the numeric value — just that it exists.)"""
    cfg = CNNConfig(weight_decay=1e-3)
    model = build_cnn(cfg)

    conv_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]

    for layer in conv_layers:
        assert layer.kernel_regularizer is not None



def test_forward_pass_batch_size():
    """This ensures it handles different batch size"""
    cfg = CNNConfig()
    model = build_cnn(cfg)

    x = tf.random.normal((8, 320, 320, 3))
    y = model(x)

    assert y.shape[0] == 8