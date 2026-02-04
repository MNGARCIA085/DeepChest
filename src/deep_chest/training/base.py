import tensorflow as tf



from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class PhaseConfig:
    lr: float = 1e-4
    epochs: int = 2
    unfreeze_layers: Optional[int] = None


@dataclass
class TrainingConfig:
    strategy: str = "standard"

    # metrics created outside ideally, but for quick tests:
    metrics: List[Any] = field(default_factory=list)

    # -------- standard --------
    lr: Optional[float] = 1e-3
    epochs: Optional[int] = 1

    # -------- transfer learning --------
    backbone_name: Optional[str] = "backbone"
    phase1: Optional[PhaseConfig] = field(default_factory=PhaseConfig)
    phase2: Optional[PhaseConfig] = field(
        default_factory=lambda: PhaseConfig(lr=1e-5, epochs=2, unfreeze_layers=1)
    )




#-----------Base trainer------------------------#
class BaseTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def compile_model(self, lr, loss, metrics):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=loss,
            metrics=metrics,
        )

    def fit_model(self, train_gen, val_gen, epochs, callbacks):
        return self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
        )



#-----------Standard trainer------------------------#
class StandardTrainer(BaseTrainer):
    def train(self, train_gen, val_gen, loss, callbacks):

        self.compile_model(
            lr=self.cfg.lr,
            loss=loss,
            metrics=self.cfg.metrics,
        )

        self.fit_model(
            train_gen,
            val_gen,
            self.cfg.epochs,
            callbacks,
        )


#-----------Transfer Learning trainer-----------------#
class TransferLearningTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)

        if not cfg.backbone_name:
            raise ValueError("backbone_name is required for transfer learning")

        self.backbone = model.get_layer(cfg.backbone_name)



    def freeze_backbone(self):
        # self.backbone.trainable = False
        # v2: safer with nested models
        for layer in self.backbone.layers:
            layer.trainable = False

    
    def unfreeze_top_layers(self, n_layers: int):
        if n_layers is None or n_layers <= 0:
            return

        layers_to_unfreeze = self.backbone.layers[-n_layers:]

        for layer in layers_to_unfreeze:
            layer.trainable = True


    def train(self, train_gen, val_gen, loss, callbacks):
        print('Feature extraction')
        self._phase1(self.cfg.phase1, train_gen, val_gen, loss, callbacks)
        print('Fine tuning')
        self._phase2(self.cfg.phase2, train_gen, val_gen, loss, callbacks)


    def _phase1(self, phase_cfg, train_gen, val_gen, loss, callbacks):
        
        self.freeze_backbone()

        print_trainable_layers(self.model) # minor test

        self.compile_model(
            lr=phase_cfg.lr,
            loss=loss,
            metrics=self.cfg.metrics,
        )

        self.fit_model(
            train_gen,
            val_gen,
            phase_cfg.epochs,
            callbacks,
        )



    def _phase2(self, phase_cfg, train_gen, val_gen, loss, callbacks):
        
        self.unfreeze_top_layers(phase_cfg.unfreeze_layers)

        print_trainable_layers(self.model) # minor test

        self.compile_model(
            lr=phase_cfg.lr,
            loss=loss,
            metrics=self.cfg.metrics,
        )

        self.fit_model(
            train_gen,
            val_gen,
            phase_cfg.epochs,
            callbacks,
        )


    """
    train()
     ├─ phase 1 → freeze → compile → fit
     └─ phase 2 → unfreeze → compile → fit
    """





"""
Unnnested backbone:
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






"""
About TL with effnet

self.backbone.layers[-1] maybe juts batch norm.....


def unfreeze_top_layers(self, n_layers):
    for layer in reversed(self.backbone.layers):
        if n_layers == 0:
            break
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            n_layers -= 1
"""




#https://chatgpt.com/c/6967ea99-3da8-832c-a919-130459f6af99


"""
for hydra
optimizer:
  _target_: tensorflow.keras.optimizers.Adam
  learning_rate: 1e-4

loss:
  _target_: myproj.losses.weighted_bce_loss
  pos_weights: ${data.pos_weights}
"""



