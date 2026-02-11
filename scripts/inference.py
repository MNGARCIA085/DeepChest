from deep_chest.preprocessors.base import DataModule
from deep_chest.core.paths import get_data_root

import hydra
from omegaconf import DictConfig

from deep_chest.infra.utils import get_best_run_id, get_top_k_run_ids
from deep_chest.infra.tracking import load_inference_bundle
from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.inference.base import Predictor




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    
    #-----------------Load model and artifacts from MLFlow------------------#

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


    #-------------------Load data (test dataset)----------------------------#
    
    # paths
    DATA_ROOT = get_data_root()
    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TRAIN_PATH = DATA_ROOT/ cfg.preprocessor.train_path
    VALID_PATH = DATA_ROOT/ cfg.preprocessor.valid_path
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path

    # appropiate prep. fn.
    prep_fn = PREP_FN_REGISTRY[model_type]

    # test generator
    data = DataModule(
        train_csv=TRAIN_PATH,
        val_csv=VALID_PATH,
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=cfg.preprocessor.labels,
        preprocess_fn=prep_fn,
    )
    data.load_data()
    test_gen = data.test_generator()
    # do i need pos weights??????

    print(test_gen)


    


    #--------------------Preds------------------------------#
    
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



    # ------------ Only one pred ------------#
    
    print('----------Pred with only 1 sample-------------')

    print(data.train_df.iloc[0].Image)

    # path
    one_path = TRAIN_PATH / data.train_df.iloc[0].Image
    one_path = "/home/marcos/Escritorio/AI-prod/DeepChest/data/nih/images-small/" + str(data.train_df.iloc[0].Image)

    # preprocess
    x = data.preprocess_image(one_path)

    # predict
    s = predictor.predict_one_probs(x)
    print(s)





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
