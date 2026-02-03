from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense121

PREP_FN_REGISTRY = {
    "simple_cnn": None,
    "cnn": None,
    "efficientnet": preprocess_input_effnet,
    "densenet": preprocess_dense121
}


"""
model_fn = PREP_FN_REGISTRY[cfg.model_key]
model = model_fn(**params)
"""