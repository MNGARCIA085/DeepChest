


# adaptate appropiately
# what if i want lo load the model from a local disk in fapi????????????????


def build_inference_pipeline(model_uri, model_type, cfg):
    model = load_model_from_mlflow(model_uri)

    prep_fn = PREP_FN_REGISTRY[model_type]

    prep = DataModule(
        labels=cfg.preprocessor.labels, # adpatet; in faspi tehy come from metadata
        preprocess_fn=prep_fn,
    )

    predictor = Predictor(model)

    return InferencePipeline(prep, predictor)
