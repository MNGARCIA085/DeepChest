

from .simple_cnn import SimpleCNN
from .efficient_net import build_efficientnet


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "efficientnet": build_efficientnet,
}


"""
model_fn = MODEL_REGISTRY[cfg.model_key]
model = model_fn(**params)
"""