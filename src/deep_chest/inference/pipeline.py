

"""
Wrapper

prep+predictor -> best later for my fastapi app



class InferencePipeline:
    def __init__(self, prep, predictor):
        self.prep = prep
        self.predictor = predictor

    def predict(self, x):
        x = self.prep.transform(x)
        return self.predictor.predict(x)
"""



"""
MAYBE

inference/
  base.py          # Predictor interface
  pipeline.py      # InferencePipeline
  loader.py        # load model + prep from artifacts
"""