import hydra
from omegaconf import DictConfig

from deep_chest.infra.config import init_mlflow
from deep_chest.core.paths import get_data_root
from deep_chest.preprocessors.base import DataModule
from deep_chest.infra.utils import get_best_run_id, get_top_k_run_ids
from deep_chest.infra.tracking import load_inference_bundle
from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.inference.base import Predictor
from deep_chest.inference.pipeline import InferencePipeline



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # 0. init mlflow
    init_mlflow(cfg.experiment_name)

    
    # 1. Load model and artifacts from MLFlow

    # best run_id
    a = get_best_run_id()
    print(a)

    """
    if i want top-k
    b = get_top_k_run_ids(k=2)
    print(b)
    """

    data = load_inference_bundle(a)

    model = data.model
    model_type = data.model_type
    labels = data.labels['labels']


    # 2. Data class
    
    # paths
    DATA_ROOT = get_data_root()
    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path_small # faster test

    # appropiate prep. fn.
    prep_fn = PREP_FN_REGISTRY[model_type] # later puit it under the prep class??????

    # test generator
    data = DataModule(
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=cfg.preprocessor.labels,
        preprocess_fn=prep_fn,
    )

    data.load_test_data()
    test_gen = data.test_generator()
    print(test_gen)


    # 3. Predictions (test generator)
    print('-------Preds with test set------------------')


    predictor = Predictor(model)

    logits = predictor.predict_logits(test_gen)
    print(logits[:3])

    probs = predictor.predict_probs(test_gen)
    print(probs[:3])

    # probs and true labels
    probs, y_true = predictor.predict_with_targets(test_gen)
    print(probs) # then i should pass through a threshold
    print(y_true)

    print('\n') 
    
   

    # 4. Predictions (1 samnple)
    print('----------Pred with only 1 sample-------------\n')

    print(data.test_df.iloc[0].Image)

    # path
    one_path = TEST_PATH / data.test_df.iloc[0].Image
    one_path = "/home/marcos/Escritorio/AI-prod/DeepChest/data/nih/images-small/" + str(data.test_df.iloc[0].Image)

    # preprocess
    x = data.preprocess(one_path)

    # predict
    s = predictor.predict_one_probs(x)
    print(s)



    # 5. Inference pipeline test (for the API)
    print('----------Inference pipeline-------------')
    # to reinforce i need to pass an instance of a prep and an instance of a predictor


    prep_fn = PREP_FN_REGISTRY[model_type]

    prep = DataModule(
        labels=cfg.preprocessor.labels,
        preprocess_fn=prep_fn,
    )

    predictor = Predictor(model)

    inference_pipeline = InferencePipeline(prep, predictor)
    pred = inference_pipeline.predict(one_path)
    print('--Path--')
    print(pred)
    print('/n')


    # simulate bytes input
    with open(one_path, "rb") as f:
        image_bytes = f.read()

    pred = inference_pipeline.predict(image_bytes)
    print('--Bytes--')
    print(pred)


    # 6. Using my builder
    print('----------Using the builder (for the API)----------------')









if __name__==main():
    main()






""""
from tensorflow import keras
LOAD A CUSTOM MODEL

    model = keras.models.load_model(
        "test_model.keras",
        compile=False, #if i dont pass loss; good for quick test; i can do this since i dont need it for inference
                    # i aslo dont need it if a compute metrics myself; but KEras do need for model.eval...
        custom_objects={"SimpleCNN": SimpleCNN},
        #custom_objects={"SimpleCNN": SimpleCNN, "loss": weighted_bce(pos_weights)}
    )

    print(model)
"""



"""
Fake bytes with numpy
import io
from PIL import Image

img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
pil_img = Image.fromarray(img)

buffer = io.BytesIO()
pil_img.save(buffer, format="PNG")
image_bytes = buffer.getvalue()
"""