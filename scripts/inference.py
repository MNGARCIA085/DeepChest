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
        compile=False, #if i dont pass loss; good for quick test; i can do this since i dont need it for inference
                    # i aslo dont need it if a compute metrics myself; but KEras do need for model.eval...
        custom_objects={"SimpleCNN": SimpleCNN},
        #custom_objects={"SimpleCNN": SimpleCNN, "loss": weighted_bce(pos_weights)}
    )

    print(model)



    #--------------------preds test--------------------------------
    predictor = Predictor(model)

    #aux = predictor.predict_logits(val_gen)
    #print(aux)

    #aux = predictor.predict_probs(val_gen)
    #print(aux)

    probs, y_true = predictor.predict_with_targets(val_gen)
    print(probs) # then i should pass through a threshold
    print(y_true)





    # ------------ 1 sample ------------#
    print(data.train_df.iloc[0].Image)

    # path
    one_path = TRAIN_PATH / data.train_df.iloc[0].Image
    one_path = "/home/marcos/Escritorio/AI-prod/DeepChest/data/nih/images-small/" + str(data.train_df.iloc[0].Image)
    print(one_path)

    # preprocess
    x = data.preprocess_image(one_path)

    # predict
    s = predictor.predict_one_probs(x)
    print(s)












if __name__==main():
    main()
