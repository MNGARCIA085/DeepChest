import tensorflow as tf




class Trainer:
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        loss: tf.keras.losses.Loss,
        metrics: list[tf.keras.metrics.Metric] | None = None,
    ):
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics or [],
        )

    def fit(
        self,
        train_gen,
        val_gen,
        epochs,
        steps_per_epoch=None,
        validation_steps=None,
        callbacks=None,
    ):
        return self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks or [],
        )









class TransferLearningTrainer(Trainer):
    def __init__(self, model, backbone_name, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.backbone = model.get_layer(backbone_name)

    def freeze_backbone(self):
        self.backbone.trainable = False

    def unfreeze_top_layers(self, n_layers):
        for layer in self.backbone.layers[-n_layers:]:
            layer.trainable = True





#https://chatgpt.com/c/6967ea99-3da8-832c-a919-130459f6af99



"""
trainer = Trainer(
    model=SimpleCNN(),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=weighted_bce_loss(pos_weights),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
        tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
    ],
)

trainer.compile()
"""


"""
for hydra
optimizer:
  _target_: tensorflow.keras.optimizers.Adam
  learning_rate: 1e-4

loss:
  _target_: myproj.losses.weighted_bce_loss
  pos_weights: ${data.pos_weights}
"""



