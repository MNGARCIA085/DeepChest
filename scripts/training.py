from deep_chest.preprocessors.base import DataModule
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense


from deep_chest.core.paths import get_data_root
from deep_chest.losses.weighted_bce import weighted_bce
from deep_chest.models.simple_cnn import SimpleCNN
from deep_chest.training.base import Trainer
from deep_chest.training.callbacks import early_stopping
import tensorflow as tf


# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")


    print(cfg)

    # params for training; not really i can use cfg.training given how i execute the script
    cfg_model = OmegaConf.load(f"config/models/{model_type}.yaml") 
    print(cfg_model)


    # paths
    DATA_ROOT = get_data_root()

    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TRAIN_PATH = DATA_ROOT/ cfg.preprocessor.train_path
    VALID_PATH = DATA_ROOT/ cfg.preprocessor.valid_path
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path



    # load data and prepare generators
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
    val_gen = data.val_generator()

    print(train_gen, val_gen)


    # model
    model = SimpleCNN(**cfg_model)

    print(model)

    _, _, pos_weights = data.compute_pos_weights_from_generator(train_gen)
    print(pos_weights)



    # 
    callbacks = []
    callbacks.append(
    early_stopping(
            monitor="val_loss",    #"val_auroc",
            patience=3,
        )
    )

    
    # train model (here, without transfer learning bc the model is not applicable for that)
    trainer = Trainer(model=model)

    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
        ],
    )

    trainer.fit(train_gen, val_gen, 1, 100, 25, callbacks)


    print(trainer.model)





if __name__==main():
    main()
