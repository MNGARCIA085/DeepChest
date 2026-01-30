from deep_chest.preprocessors.base import DataModule
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense

from deep_chest.core.paths import get_data_root
from deep_chest.models.simple_cnn import SimpleCNN
import tensorflow as tf

# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf

from tensorflow import keras
from deep_chest.inference.base import Predictor


import numpy as np




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")


    print(cfg)

    # params for training; not really i can use cfg.training given how i execute the script
    cfg_model = OmegaConf.load(f"config/models/{model_type}.yaml") # NOT use this!!!!
    print(cfg_model)


    # paths
    DATA_ROOT = get_data_root()

    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TRAIN_PATH = DATA_ROOT/ cfg.preprocessor.train_path
    VALID_PATH = DATA_ROOT/ cfg.preprocessor.valid_path
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path



    # load data and prepare generators

    # later -> my registry for prep. fn.

    data = DataModule(
        train_csv=TRAIN_PATH,
        val_csv=VALID_PATH,
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=cfg.preprocessor.labels,
        preprocess_fn=None, # add prep according to model type
    )
    data.load_data()
    train_gen = data.train_generator()
    val_gen = data.val_generator() # here: trick to show how to make inference; maybe  fake create ds.
                # same seed for same val_gen

    print(train_gen, val_gen)

    

    # load model
    model = keras.models.load_model(
        "test_model.keras",
        compile=False, 
        custom_objects={"SimpleCNN": SimpleCNN},
    )

    print(model)



    #-------------------Preds and true labels------------------#
    predictor = Predictor(model)
    y_preds_proba, y_true = predictor.predict_with_targets(val_gen)


    #----------------------Evaluation---------------------------#
    from deep_chest.evaluation.evaluator import Evaluator

    labels = ['Cardiomegaly',
                     'Emphysema',
                     'Effusion',
                     'Hernia',
                     'Infiltration',
                     'Mass',
                     'Nodule',
                     'Atelectasis',
                     'Pneumothorax',
                     'Pleural_Thickening',
                     'Pneumonia',
                     'Fibrosis',
                     'Edema',
                     'Consolidation']
    thresholds = np.array([0.3, 0.5, 0.4, 0.4, 0.3, 0.5, 0.4, 0.4,0.3, 0.5, 0.4, 0.4,0.1, 0.2])

    evaluator = Evaluator(thresholds, labels)
    results = evaluator.evaluate(y_true, y_preds_proba)    

    print(results)

    #plot = evaluator.precision_recall_curve(y_true, y_preds_proba)

    #ci = evaluator.bootstrap_auc(y_true, y_preds_proba)
    #print(ci)

    plot = evaluator.plot_calibration_curve(y_true, y_preds_proba)

















if __name__==main():
    main()
