class BaseTrainer:
    def __init__(self, model, loss, optimizer, metrics):
        self.model = model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def fit(self, train_gen, val_gen, **kwargs):
        return self.model.fit(
            train_gen,
            validation_data=val_gen,
            **kwargs
        )

    def predict_logits(self, x):
        return self.model(x, training=False)

    def predict_proba(self, x):
        return tf.sigmoid(self.predict_logits(x))


class TransferLearningTrainer(BaseTrainer):
    def __init__(self, model, backbone_name, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.backbone = model.get_layer(backbone_name)

    def freeze_backbone(self):
        self.backbone.trainable = False

    def unfreeze_top_layers(self, n_layers):
        for layer in self.backbone.layers[-n_layers:]:
            layer.trainable = True

trainer = TransferLearningTrainer(
    model=model,
    backbone_name="efficientnetb0",
    loss=loss,
    optimizer=optimizer,
    metrics=metrics
)


"""

ver bien cómo se usa

trainer.train..........
trainer.unfreeze....
trainer.train...





trainer = TransferLearningTrainer(
    model=model,
    backbone_name="efficientnetb0",
    loss=weighted_bce_loss(pos_weights),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", multi_label=True),
        tf.keras.metrics.AUC(curve="PR", multi_label=True),
    ]
)



freeze backbone before training
trainer.freeze_backbone()
trainer.compile()  # REQUIRED after changing trainable flags


trainer.freeze_backbone()
trainer.compile()  # REQUIRED after changing trainable flags


--
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
❌ Mixing inference logic in the traine


predictor = Predictor(model, thresholds=thresholds)
probs = predictor.predict_proba(x)



"""