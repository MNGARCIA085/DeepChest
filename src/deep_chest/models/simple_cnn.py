import tensorflow as tf
from tensorflow.keras import layers


class SimpleCNN(tf.keras.Model):
    def __init__(
        self,
        input_shape=(320, 320, 3),
        num_classes=14,
        conv_filters=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        use_pooling=(True, True, False),
        pool_sizes=(2, 2, 2),
        dense_units=128,
        dropout_rate=0.5,
        use_batchnorm=True,
        name="simple_cnn_baseline",
    ):
        super().__init__(name=name)

        # ---- validation (fail fast) ----
        n = len(conv_filters)
        assert len(kernel_sizes) == n
        assert len(use_pooling) == n
        assert len(pool_sizes) == n

        # ---- convolutional trunk ----
        self.convs = []
        self.bns = []
        self.pools = []

        for f, k, do_pool, p in zip(
            conv_filters, kernel_sizes, use_pooling, pool_sizes
        ):
            self.convs.append(
                layers.Conv2D(
                    filters=f,
                    kernel_size=k,
                    padding="same",
                )
            )

            self.bns.append(
                layers.BatchNormalization() if use_batchnorm else None
            )

            if do_pool:
                self.pools.append(
                    layers.MaxPooling2D(pool_size=(p, p))
                )
            else:
                self.pools.append(None)

        # ---- head ----
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(dense_units, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes)  # logits


    def call(self, inputs, training=None):
        x = inputs

        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = conv(x)
            if bn is not None:
                x = bn(x, training=training)
            x = tf.nn.relu(x)
            if pool is not None:
                x = pool(x)

        x = self.gap(x)
        x = self.fc(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)




