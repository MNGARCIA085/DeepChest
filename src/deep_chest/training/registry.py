from .base import StandardTrainer, TransferLearningTrainer



TRAINER_REGISTRY = {
    "standard": StandardTrainer,
    "transfer_learning": TransferLearningTrainer,
}





