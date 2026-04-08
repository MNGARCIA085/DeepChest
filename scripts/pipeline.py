import hydra
from omegaconf import DictConfig

from deep_chest.pipelines.training_pipeline import TrainingPipeline


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pipeline = TrainingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()