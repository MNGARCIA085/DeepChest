import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)

class DataModule:
    def __init__(
        self,
        train_csv,
        val_csv,
        test_csv,
        image_dir,
        labels,
        x_col="Image",
        batch_size=8,
        image_size=(320, 320),
        preprocess_fn=None,
        seed=1,
    ):
        self.train_csv = train_csv # path
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.image_dir = image_dir
        self.labels = labels
        self.x_col = x_col
        self.batch_size = batch_size
        self.image_size = image_size
        self.preprocess_fn = preprocess_fn
        self.seed = seed

        self.train_df = None
        self.val_df = None
        self.test_df = None


        self.train_gen = None # light; ImageDataGenerator.flow_from_dataframe does not load images into memory.
        self.val_gen = None
        self.test_gen = None
        self.pos_weights = None
        self.pos_freq = None
        self.neg_freq = None





    # ---------- Data loading ----------#
    def load_data(self):
        self.train_df = pd.read_csv(self.train_csv)
        self.val_df = pd.read_csv(self.val_csv)
        self.test_df = pd.read_csv(self.test_csv)

        return self.train_df, self.val_df, self.test_df



    # ---------- Generators ----------
    def _make_image_generator(self, shuffle):
        if self.preprocess_fn is None:
            return ImageDataGenerator(rescale=1.0 / 255)
        return ImageDataGenerator(preprocessing_function=self.preprocess_fn)

    def _flow(self, df, shuffle):
        datagen = self._make_image_generator(shuffle)

        return datagen.flow_from_dataframe(
            dataframe=df,
            directory=self.image_dir,
            x_col=self.x_col,
            y_col=self.labels,
            class_mode="raw",
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
            target_size=self.image_size,
        )

    def train_generator(self):
        if self.train_df is None:
            raise RuntimeError("Call load_data() first")
        return self._flow(self.train_df, shuffle=True)

    def val_generator(self):
        if self.val_df is None:
            raise RuntimeError("Call load_data() first")
        return self._flow(self.val_df, shuffle=False)

    def test_generator(self):
        if self.test_df is None:
            raise RuntimeError("Call load_data() first")
        return self._flow(self.test_df, shuffle=False)

    
    # ---------- Single-image preprocessing (inference) ----------
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.image_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if self.preprocess_fn is None:
            x = x / 255.0
        else:
            x = self.preprocess_fn(x)

        return x

    # ---------- Logging / experiment tracking ----------
    def get_artifacts(self):
        base = {
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_labels": len(self.labels),
            "labels": self.labels,
            "preprocessing": (
                "rescale_1_255" if self.preprocess_fn is None
                else self.preprocess_fn.__name__
            ),
            "seed": self.seed,
        }

        if hasattr(self, "pos_weights") and self.pos_weights is not None:
            base["pos_weights"] = self.pos_weights

        return base



    #-----------------for weighting the loss------------------
    @staticmethod
    def compute_pos_weights_from_generator(gen):
        labels = gen.labels
        N = labels.shape[0]

        pos_freq = labels.sum(axis=0) / N
        neg_freq = 1 - pos_freq


        pw = neg_freq / pos_freq
        return pw, pos_freq, neg_freq
    # check calculations later!!!!

    

    # 
    def prepare_training(self):
        self.load_data()

        self.train_gen = self.train_generator()
        self.val_gen = self.val_generator()

        # computed with train data
        pw, pos_f, neg_f = self.compute_pos_weights_from_generator(self.train_gen)


        self.pos_weights = pw
        self.pos_freq = pos_f
        self.neg_freq = neg_f












"""
celaner option for later
Instead of prepare_training, many projects use:

setup(stage="train")

build()

initialize()

Example:

data.setup("train")


Later you could do:

data.setup("inference")


and skip weights.
"""




"""

data = ChestXrayDataModule(
    train_csv=TRAIN_PATH,
    val_csv=VALID_PATH,
    test_csv=TEST_PATH,
    image_dir=IMAGE_DIR,
    labels=labels,
    preprocess_fn=None,
)
data.load_data()
train_gen = data.train_generator()


from tensorflow.keras.applications.efficientnet import preprocess_input

data = ChestXrayDataModule(
    ...,
    preprocess_fn=preprocess_input,
)

from tensorflow.keras.applications.densenet import preprocess_input

data = ChestXrayDataModule(
    ...,
    preprocess_fn=preprocess_input,
)
"""






"""
# Simple CNN
train_gen = get_generator(
    df_train,
    image_dir,
    x_col,
    y_cols,
    preprocess_fn=None
)

# EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input
train_gen = get_generator(
    df_train,
    image_dir,
    x_col,
    y_cols,
    preprocess_fn=preprocess_input
)

# DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
train_gen = get_generator(
    df_train,
    image_dir,
    x_col,
    y_cols,
    preprocess_fn=preprocess_input
)
"""