
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet

PREP_FN_REGISTRY = {
    "simple_cnn": None,
    "efficientnet": preprocess_input_effnet,
}


"""
model_fn = MODEL_REGISTRY[cfg.model_key]
model = model_fn(**params)
"""