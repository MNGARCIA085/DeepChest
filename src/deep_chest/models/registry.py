

from .cnn import CNNConfig, build_cnn
from .efficient_net import EfficientNetConfig, build_efficientnet
from .densenet121 import DenseNetConfig, build_densenet


# model registry includes config + build


MODEL_REGISTRY = {
    "cnn": (CNNConfig, build_cnn),
    "efficientnet": (EfficientNetConfig, build_efficientnet),
    "densenet": (DenseNetConfig, build_densenet),
}

"""

"""



"""
MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "efficientnet": build_efficientnet,
}
"""

"""
model_fn = MODEL_REGISTRY[cfg.model_key]
model = model_fn(**params)
"""