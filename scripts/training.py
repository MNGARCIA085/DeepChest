import tensorflow as tf
from deep_chest.core.paths import get_data_root
from deep_chest.preprocessors.base import DataModule
from deep_chest.losses.weighted_bce import weighted_bce
from deep_chest.training.base import Trainer
from deep_chest.training.callbacks import early_stopping
from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.models.registry import MODEL_REGISTRY


# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig


from deep_chest.training.base import TransferLearningTrainer


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type.name
    print(f"\nSelected model: {model_type}")


    print(cfg)



    #--------Loading data and preprocessing-----------------------#
    
    # paths
    DATA_ROOT = get_data_root()
    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TRAIN_PATH = DATA_ROOT/ cfg.preprocessor.train_path
    VALID_PATH = DATA_ROOT/ cfg.preprocessor.valid_path
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path

    # prep. fn
    prep_fn = PREP_FN_REGISTRY[model_type]
    print(prep_fn)


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



    # choose model
    cfg_class, build_model = MODEL_REGISTRY[model_type]
    model_cfg = cfg_class(**cfg.model_type.models)
    model = build_model(model_cfg)

    print(model.summary())


    #----------Calculations required---------------------#
    _, _, pos_weights = data.compute_pos_weights_from_generator(train_gen)
    print(pos_weights)



    #----------------Training--------------------------------

    # Callbacks
    callbacks = []
    callbacks.append(
    early_stopping(
            monitor="val_loss", #"val_auroc",
            patience=3,
        )
    )


    # train model : without TL, let's see later how to combine
    trainer = Trainer(model=model)

    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.model_type.training.lr),
        loss=weighted_bce(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
        ],
    )

    # see how to calculate val_steps.....
    trainer.fit(train_gen, val_gen, cfg.model_type.training.epochs, 100, 25, callbacks)











    
    


    

if __name__==main():
    main()
