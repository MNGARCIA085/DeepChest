import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)





class DataModule:
    def __init__(
        self,
        train_csv=None,
        val_csv=None,
        test_csv=None,
        image_dir=None,
        labels=[],
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
    def load_train_data(self):
        self.train_df = pd.read_csv(self.train_csv)


    def load_val_data(self):
        self.val_df = pd.read_csv(self.val_csv)


    def load_test_data(self):
        self.test_df = pd.read_csv(self.test_csv)



    # ---------- Generators -----------------------#
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
            raise RuntimeError("Call load_train_data() first")
        return self._flow(self.train_df, shuffle=True)

    def val_generator(self):
        if self.val_df is None:
            raise RuntimeError("Call load_val_data() first")
        return self._flow(self.val_df, shuffle=False)

    def test_generator(self):
        if self.test_df is None:
            raise RuntimeError("Call load_test_data() first")
        return self._flow(self.test_df, shuffle=False)

    
    #-----------------For weighting the loss------------------#
    @staticmethod
    def compute_pos_weights_from_generator(gen):
        labels = gen.labels
        N = labels.shape[0]

        pos_freq = labels.sum(axis=0) / N
        neg_freq = 1 - pos_freq


        pw = neg_freq / pos_freq
        return pw, pos_freq, neg_freq


    #----------------Prepare training---------------------------# 
    def prepare_training(self):
        self.load_train_data()
        self.load_val_data()

        self.train_gen = self.train_generator()
        self.val_gen = self.val_generator()

        # computed with train data
        pw, pos_f, neg_f = self.compute_pos_weights_from_generator(self.train_gen)


        self.pos_weights = pw
        self.pos_freq = pos_f
        self.neg_freq = neg_f


    # ---------- Logging / experiment tracking ------------------#
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



    # doc later------------------------
    #---------To receive any kind of input for the API (path, bytes..)------#
    #----------------



    # ---------- Single-image preprocessing (inference) ----------#

    # pure prep; no path
    def preprocess_array(self, img_array):
        """
        img_array: np.ndarray (H, W, C) in uint8 or float32
        Works for:
            API uploads
            DB blobs
            Webcam frames
            Anything
        """

        if img_array.shape[-1] == 4:  # RGBA; load_img is RGB, but PIL/bytes/arrays might not be
            img_array = img_array[..., :3]


        img = tf.image.resize(img_array, self.image_size)
        x = tf.cast(img, tf.float32)
        x = tf.expand_dims(x, axis=0)

        if self.preprocess_fn is None:
            x = x / 255.0
        else:
            x = self.preprocess_fn(x)

        return x


    # source adapters
    

    # from path
    def load_from_path(self, image_path):
        img = load_img(image_path)
        return img_to_array(img)


    # from bytes (for the API)
    def load_from_bytes(self, image_bytes):
        img = tf.io.decode_image(image_bytes, channels=3)
        return img.numpy()
    """
    safer version:
    def load_from_bytes(self, image_bytes):
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return img.numpy()
    """

    # from PIL
    def load_from_pil(self, pil_img):
        return np.array(pil_img)

    # from array
    def load_from_array(self, arr):
        return arr

    # dispatcher
    def select_loader(self, x):
        if isinstance(x, str):
            return self.load_from_path(x)
        elif isinstance(x, bytes):
            return self.load_from_bytes(x)
        elif isinstance(x, np.ndarray):
            return self.load_from_array(x)
        elif isinstance(x, Image.Image):
            return self.load_from_pil(x)
        else:
            raise ValueError("Unsupported input type")

    # preprocess
    def preprocess(self, x):
        data = self.select_loader(x)
        return self.preprocess_array(data)








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
for one sample preds

raw_input
   ↓
select_loader
   ↓
array (H,W,C)
   ↓
preprocess_array
   ↓
tensor (1,H,W,C)
   ↓
predictor

"""