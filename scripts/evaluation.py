from deep_chest.preprocessors.base import DataModule
from deep_chest.core.paths import get_data_root

import hydra
from omegaconf import DictConfig


from deep_chest.infra.config import init_mlflow
from deep_chest.infra.utils import get_best_run_id, get_top_k_run_ids
from deep_chest.infra.tracking import load_inference_bundle
from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.inference.base import Predictor

from deep_chest.evaluation.evaluator import Evaluator

import numpy as np


from deep_chest.infra.logging import log_test_results



import ast













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

    #convert to list
    labels_list = ast.literal_eval(labels)



    # 2. Data class
    DATA_ROOT = get_data_root()
    IMAGE_DIR = DATA_ROOT/ cfg.preprocessor.image_dir
    TEST_PATH = DATA_ROOT/ cfg.preprocessor.test_path

    prep_fn = PREP_FN_REGISTRY[model_type]

    data = DataModule(
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=labels_list,
        preprocess_fn=prep_fn,
    )

    data.load_test_data()
    test_gen = data.test_generator()


    # 3. Predictions (test generator)
    predictor = Predictor(model)
    y_preds_proba, y_true = predictor.predict_with_targets(test_gen)


    # 4. Evaluation

    # example of thresholds
    thresholds = np.array([0.3, 0.5, 0.4, 0.4, 0.3, 0.5, 0.4, 0.4,0.3, 0.5, 0.4, 0.4,0.1, 0.2])

    evaluator = Evaluator(thresholds, labels_list) # to labels later
    per_class_metrics, aggregate_metrics = evaluator.evaluate(y_true, y_preds_proba)  



    print(per_class_metrics)
    print(aggregate_metrics)
    

    #plot = evaluator.precision_recall_curve(y_true, y_preds_proba)


    curves = evaluator.get_precision_recall_curves(y_true, y_preds_proba)
    print(curves)




    ci = evaluator.bootstrap_auc(y_true, y_preds_proba)
    print(ci)

    #plot = evaluator.plot_calibration_curve(y_true, y_preds_proba)


    

    log_test_results(model_type, per_class_metrics, aggregate_metrics, curves, ci)








if __name__==main():
    main()






