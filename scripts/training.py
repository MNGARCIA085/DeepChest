import tensorflow as tf
from deep_chest.core.paths import get_data_root
from deep_chest.preprocessors.base import DataModule
from deep_chest.losses.weighted_bce import weighted_bce
from deep_chest.training.callbacks import early_stopping
from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.models.registry import MODEL_REGISTRY
from deep_chest.training.registry import TRAINER_REGISTRY

# Hydra + OmegaConf
import hydra
from omegaconf import DictConfig

from deep_chest.training.base import PhaseConfig, TrainingConfig

from deep_chest.infra.config import init_mlflow
from deep_chest.infra.logging import logging




@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):


    # init mlflow
    init_mlflow(cfg.experiment_name)

    # get model type (nn, tree.....)
    model_type = cfg.model_type.name
    print(f"\nSelected model: {model_type}")


    #print(cfg)



    #--------Loading data and preprocessing-----------------------#
    
    # paths
    DATA_ROOT = get_data_root()
    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TRAIN_PATH = DATA_ROOT/ cfg.preprocessor.train_path
    VALID_PATH = DATA_ROOT/ cfg.preprocessor.valid_path
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path

    # prep. fn
    prep_fn = PREP_FN_REGISTRY[model_type]


    # load data and prepare generators
    data = DataModule(
        train_csv=TRAIN_PATH,
        val_csv=VALID_PATH,
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=cfg.preprocessor.labels,
        preprocess_fn=prep_fn, # add prep according to model type
    )
    data.load_data()
    train_gen = data.train_generator()
    val_gen = data.val_generator()

    


    #----------Calculations required---------------------#
    _, _, pos_weights = data.compute_pos_weights_from_generator(train_gen)
    #print(pos_weights)


    # for logging
    artifacts = data.get_artifacts()


    # choose model
    cfg_class, build_model = MODEL_REGISTRY[model_type]
    model_cfg = cfg_class(**cfg.model_type.models)
    model = build_model(model_cfg)

    #print(model.summary())


    


    #----------------Training--------------------------------

    # training cfg
    training_cfg = TrainingConfig(**cfg.model_type.training)
    metrics=[
                tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
                tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
            ] # maybe a build_metrics(num_clasess...) is better
    training_cfg.metrics = metrics


    # custom loss
    loss = weighted_bce(pos_weights)

    # callbacks
    callbacks = []
    callbacks.append(
    early_stopping(
            monitor="val_loss", #"val_auroc",
            patience=3,
        )
    )


    # trainer class
    trainer_cls = TRAINER_REGISTRY[cfg.model_type.training.strategy]

    # instanciate trainer class
    trainer = trainer_cls(model, training_cfg)
    
    # train
    results = trainer.train(train_gen, val_gen, loss, callbacks)


    print('results', results)


    

    #-----------Preds (simple test)-----------------#
    from deep_chest.inference.base import Predictor

    predictor = Predictor(model)

    probs, y_true = predictor.predict_with_targets(val_gen)
    #print(probs) # then i should pass through a threshold
    #print(y_true)

    #print(probs.shape)


    #-----------Evaluation tests----------------------#
    from deep_chest.evaluation.evaluator import Evaluator

    evaluator = Evaluator(0.5, cfg.preprocessor.labels)

    metrics = evaluator.evaluate(y_true, probs)
    print(metrics)



    #----------------Logging---------------------#

    results['metrics'] = metrics
    print(results['metrics'])

    logging('training', artifacts, results, model_type, 'train')



if __name__==main():
    main()
