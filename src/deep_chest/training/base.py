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

        history = self.fit_model(
            train_gen,
            val_gen,
            self.cfg.epochs,
            callbacks,
        )


        return {
                "history": history.history,
                "hyperparams": {
                    "lr": self.cfg.lr,
                    "epochs": self.cfg.epochs,
                }
            }







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
        h1, lr1, epochs1 = self._phase1(self.cfg.phase1, train_gen, val_gen, loss, callbacks)
        print('Fine tuning')
        h2, lr2, epochs2, num_unfrozen = self._phase2(self.cfg.phase2, train_gen, val_gen, loss, callbacks)


        return {
          "phase1": {
              "history": h1.history,
              "lr": lr1,
              "epochs": epochs1,
          },
          "phase2": {
              "history": h2.history,
              "lr": lr2,
              "epochs": epochs2,
              "num_unfrozen": num_unfrozen,
          },
          #"backbone": "resnet50"
        }



    def _phase1(self, phase_cfg, train_gen, val_gen, loss, callbacks):
        
        self.freeze_backbone()

        #print_trainable_layers(self.model) # minor test

        self.compile_model(
            lr=phase_cfg.lr,
            loss=loss,
            metrics=self.cfg.metrics,
        )

        history = self.fit_model(
            train_gen,
            val_gen,
            phase_cfg.epochs,
            callbacks,
        )

        return history, phase_cfg.lr, phase_cfg.epochs



    def _phase2(self, phase_cfg, train_gen, val_gen, loss, callbacks):
        
        self.unfreeze_top_layers(phase_cfg.unfreeze_layers)

        #print_trainable_layers(self.model) # minor test

        self.compile_model(
            lr=phase_cfg.lr,
            loss=loss,
            metrics=self.cfg.metrics,
        )

        history = self.fit_model(
            train_gen,
            val_gen,
            phase_cfg.epochs,
            callbacks,
        )

        return history, phase_cfg.lr, phase_cfg.epochs, phase_cfg.unfreeze_layers


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





"""
phase 1

......


phase 2->Z FT
trainer.unfreeze_top_layers(n_layers=30)

change optim with lowe rlr
trainer.optimizer = tf.keras.optimizers.Adam(1e-5)
trainer.compile()  # REQUIRED


train again
history_ft = trainer.fit(
    train_generator,
    val_generator,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=25
)

✅ One trainer instance
✅ Same model instance
✅ Freeze → recompile → train
✅ Unfreeze → recompile → train
✅ No prediction logic in trainer

This is exactly how you’d do it in a real ML system.

6️⃣ Common mistakes (avoid these)

❌ Forgetting to recompile after trainable changes
❌ Creating a new trainer for fine-tuning
❌ Freezing inside the model builder
❌ Mixing inference logic in the trainer

"""