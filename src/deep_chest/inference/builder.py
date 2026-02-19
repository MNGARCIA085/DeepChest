import json

from keras.models import load_model

from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.preprocessors.base import DataModule
from deep_chest.inference.base import Predictor
from deep_chest.inference.pipeline import InferencePipeline
from deep_chest.infra.tracking import load_inference_bundle




# Low level builder, pure, no MLFlow
def build_pipeline_from_components(model, model_type, labels):
    prep_fn = PREP_FN_REGISTRY[model_type]

    prep = DataModule(
        labels=labels,
        preprocess_fn=prep_fn,
    )

    predictor = Predictor(model)

    return InferencePipeline(prep, predictor)



# MLFlow loader
def build_pipeline_from_run(run_id):
    data = load_inference_bundle(run_id)

    return build_pipeline_from_components(
        model=data.model,
        model_type=data.model_type,
        labels=data.labels,
    )


# for my API (load a model from a path and its metadata)
def build_pipeline_from_export(export_path):
    #model = tf.keras.models.load_model(export_path / "model", compile=False)
    model = load_model(export_path / "model/model.keras", compile=False)

    with open(export_path / "metadata.json") as f:
        meta = json.load(f)

    return build_pipeline_from_components(
        model=model,
        model_type=meta["model_type"],
        labels=meta["labels"],
    )




"""
In FastAPI:

export/
    model/
        saved_model/
    metadata.json

{
  "model_type": "resnet50",
  "labels": ["Atelectasis", ...]
}



pipeline = build_pipeline_from_export("/models/chest_xray_v1")


“Give me a pipeline object that has a .predict() method.”

"""



"""
Training world (MLflow)
        ↓
Export artifact
        ↓
ML package loads artifact
        ↓
FastAPI consumes ML package
"""