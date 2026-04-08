import tensorflow as tf

from deep_chest.infra.config import init_mlflow
from deep_chest.core.paths import get_data_root
from deep_chest.preprocessors.base import DataModule
from deep_chest.preprocessors.registry import PREP_FN_REGISTRY
from deep_chest.models.registry import MODEL_REGISTRY
from deep_chest.losses.weighted_bce import weighted_bce
from deep_chest.training.callbacks import early_stopping
from deep_chest.training.base import TrainingConfig
from deep_chest.training.registry import TRAINER_REGISTRY
from deep_chest.infra.logging import logging

from deep_chest.inference.base import Predictor
from deep_chest.evaluation.evaluator import Evaluator


class TrainingPipeline:

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.data = None
        self.trainer = None
        self.results = {}

    # ---------------- Setup ---------------- #

    def setup(self):
        init_mlflow(self.cfg.experiment_name)

        self.model_type = self.cfg.model_type.name
        print(f"\nSelected model: {self.model_type}")
        print(self.cfg)

        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_aux()


    def _setup_data(self):
        DATA_ROOT = get_data_root()

        IMAGE_DIR = DATA_ROOT / self.cfg.preprocessor.image_dir
        TRAIN_PATH = DATA_ROOT / self.cfg.preprocessor.train_path
        VALID_PATH = DATA_ROOT / self.cfg.preprocessor.valid_path
        TEST_PATH = DATA_ROOT / self.cfg.preprocessor.test_path

        prep_fn = PREP_FN_REGISTRY[self.model_type]

        self.data = DataModule(
            train_csv=TRAIN_PATH,
            val_csv=VALID_PATH,
            test_csv=TEST_PATH,
            image_dir=IMAGE_DIR,
            labels=self.cfg.preprocessor.labels,
            preprocess_fn=prep_fn,
        )

        self.data.prepare_training()
        self.artifacts = self.data.get_artifacts()


    def _setup_model(self):
        cfg_class, build_model = MODEL_REGISTRY[self.model_type]

        model_cfg = cfg_class(**self.cfg.model_type.models)
        self.model_cfg = model_cfg

        self.model = build_model(model_cfg)


    def _setup_training(self):
        training_cfg = TrainingConfig(**self.cfg.model_type.training)

        metrics = [
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
        ]

        training_cfg.metrics = metrics
        self.training_cfg = training_cfg

        trainer_cls = TRAINER_REGISTRY[self.cfg.model_type.training.strategy]
        self.trainer = trainer_cls(self.model, training_cfg)


    def _setup_aux(self):
        self.loss = weighted_bce(self.data.pos_weights)

        self.callbacks = [
            early_stopping(
                monitor="val_loss",
                patience=3,
            )
        ]

        self.predictor = Predictor(self.model)

        self.evaluator = Evaluator(
            threshold=0.5,
            labels=self.cfg.preprocessor.labels,
        )

    # ---------------- Core Steps ---------------- #

    def train(self):
        self.results = self.trainer.train(
            self.data.train_gen,
            self.data.val_gen,
            self.loss,
            self.callbacks,
        )

    def evaluate(self):
        probs, y_true = self.predictor.predict_with_targets(self.data.val_gen)

        per_class_metrics, agg_metrics = self.evaluator.evaluate(y_true, probs)

        self.results["metrics"] = per_class_metrics
        self.results["agg_metrics"] = agg_metrics

    def log(self):
        if self.cfg.tracking.enabled:
            logging(
                "training",
                self.artifacts,
                self.results,
                self.model,
                self.model_cfg,
                self.model_type,
                "train",
            )
        else:
            print(self.results["metrics"])
            print(self.results["agg_metrics"])

    # ---------------- Run ---------------- #

    def run(self):
        self.setup()
        self.train()
        self.evaluate()
        self.log()
