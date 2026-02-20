

class InferencePipeline:
    def __init__(self, prep, predictor):
        self.prep = prep
        self.predictor = predictor

    def predict(self, x):
        data = self.prep.preprocess(x)
        return self.predictor.predict_one_probs(data)


"""
MAYBE

inference/
  base.py          # Predictor interface
  pipeline.py      # InferencePipeline
  loader.py        # load model + prep from artifacts
"""