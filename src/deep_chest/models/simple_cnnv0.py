import tensorflow as tf
from tensorflow.keras import layers


class SimpleCNN(tf.keras.Model):
    def __init__(self, input_shape=(320, 320, 3), num_classes=14, name="simple_cnn_baseline"):
        super().__init__(name=name)

        #self.input_layer = layers.InputLayer(input_shape=input_shape)

        self.conv1 = layers.Conv2D(32, 3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D()

        self.conv2 = layers.Conv2D(64, 3, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D()

        self.conv3 = layers.Conv2D(128, 3, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.gap = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.5)

        self.classifier = layers.Dense(num_classes)  # logits

    def call(self, inputs, training=False):
        x = inputs

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.gap(x)

        x = self.fc(x)
        x = self.dropout(x, training=training)

        return self.classifier(x)



#model = SimpleCNN(input_shape=(320, 320, 3), num_classes=14)



"""
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=weighted_bce_loss(pos_weights),
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
        tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
    ]
)

model.summary()
"""

"""
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,
    steps_per_epoch=100,
    validation_steps=25
)
"""