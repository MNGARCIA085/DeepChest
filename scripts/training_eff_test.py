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


from deep_chest.training.base import TransferLearningTrainer


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # get model type (nn, tree.....)
    model_type = cfg.model_type
    print(f"\nSelected model: {model_type}")


    print(cfg)

    # params for training; not really i can use cfg.training given how i execute the script; NOT!!!!!
    #cfg_model = OmegaConf.load(f"config/models/{model_type}.yaml") 
    #print(cfg_model)


    

    from deep_chest.preprocessors.registry import PREP_FN_REGISTRY

    prep_fn = PREP_FN_REGISTRY['efficientnet']
    print(prep_fn)



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



    # choose model
    from deep_chest.models.registry import MODEL_REGISTRY
    from deep_chest.models.efficient_net import EfficientNetConfig 

    model_fn = MODEL_REGISTRY['efficientnet'] # not hardcode later
    cfg = EfficientNetConfig # later overwrite with hydra
    model = model_fn(cfg)

  
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



    #-----------------------------


    # test freeze/unfreeze


    """

    for an unnnested backbone
    for l in trainer.backbone.layers[-10:]:
    print(l.name, l.trainable)
    """

    # need like this cauise my backbone is nested
    def print_trainable_layers(model, indent=0):
        prefix = " " * indent
        print(f"{prefix}{model.name} (Model) - trainable={model.trainable}")

        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                print_trainable_layers(layer, indent + 2)
            else:
                print(
                    f"{' ' * (indent + 2)}{layer.name:30} | "
                    f"{layer.__class__.__name__:20} | "
                    f"trainable={layer.trainable}"
                )





    trainer = TransferLearningTrainer(model=model, backbone_name="backbone") 

    print_trainable_layers(model)


    trainer.freeze_backbone()

    print_trainable_layers(model)

    trainer.unfreeze_top_layers(n_layers=5)


    print_trainable_layers(model)



    return





    # --------------- Training (with TL) ------------------------- #

    

    trainer = TransferLearningTrainer(model=model, backbone_name="backbone") # quitar hardcdoe later


    # phase 1
    trainer.freeze_backbone()

    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
        ]
    )

    hiistiry = trainer.fit(
        train_gen,
        val_gen,
        epochs=2,
        steps_per_epoch=100,
        validation_steps=25
    )




    #------Phase 2-----------

    # unfreeze some layers
    trainer.unfreeze_top_layers(n_layers=1)

    # recompile!!!!!!
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5), # lower LR
        loss=weighted_bce(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
        ])


    # train again
    history_ft = trainer.fit(
        train_gen,
        val_gen,
        epochs=2,
        steps_per_epoch=100,
        validation_steps=25
    )



    
    


    



    













if __name__==main():
    main()
