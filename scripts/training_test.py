# for a quick colab test
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


import tensorflow as tf

import pandas as pd





from pathlib import Path
import os

def get_data_root() -> Path:
    try:
        return Path(os.environ["DATA_ROOT"]).resolve()
    except KeyError:
        raise RuntimeError(
            "DATA_ROOT is not set. "
            "Set it via environment variable."
        )




print(get_data_root())



#IMAGE_DIR = "data/nih/images-small/"
# ver lo de nih; lo inclui en la variable de entorno


DATA_ROOT = get_data_root()

IMAGE_DIR = DATA_ROOT/"images-small/"
TRAIN_PATH = DATA_ROOT/ "train-small.csv"
VALID_PATH = DATA_ROOT/ "valid-small.csv"
TEST_PATH = DATA_ROOT/ "test.csv"








# adapt appropiately in colab (but firsd, local test)
def load_data():
    """
    train_df = pd.read_csv("data/nih/train-small.csv")
    valid_df = pd.read_csv("data/nih/valid-small.csv")
    test_df = pd.read_csv("data/nih/test.csv")
    """
    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, valid_df, test_df



# labels
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']




# prevent data leakegae......




def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    return generator




def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, 
                                    batch_size=8, seed=1, target_w = 320, target_h = 320):



    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir, #IMAGE_DIR
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    #batch = raw_train_generator.next()
    batch = next(raw_train_generator)
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator





def create_generators(train_df, valid_df, test_df, labels): # labels....
    #IMAGE_DIR = "data/nih/images-small/"
    train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
    valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)
    return train_generator, valid_generator, test_generator


# class imbalance................



def create_model():
    # create the base pre-trained model
    #base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)
    base_model = DenseNet121(include_top=False)

    x = base_model.output

    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    #model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy())

    return model




def main():


    # load data
    train_df, valid_df, test_df = load_data()

    print(train_df.head())


    # create generators
    train_generator, valid_generator, test_generator = create_generators(train_df, valid_df, test_df, labels)

    # create model
    model = create_model()


    



if __name__==main():
    main()



#sed -i 's/\xC2\xA0/ /g; s/\t/    /g' scripts/training_test.py



"""
history = model.fit(train_generator, 
      validation_data=valid_generator,
      steps_per_epoch=100, 
      validation_steps=25, 
      epochs = 1)
"""





"""
Data root via env. variable




console -> export DATA_ROOT=/absolute/path/to/data/my_project
console -> export DATA_ROOT=/home/marcos/Escritorio/AI-prod/DeepChest/data/nih




from pathlib import Path
import os

def get_data_root() -> Path:
    try:
        return Path(os.environ["DATA_ROOT"]).resolve()
    except KeyError:
        raise RuntimeError(
            "DATA_ROOT is not set. "
            "Set it via environment variable."
        )


everywhere else:

DATA_ROOT = get_data_root()
train_path = DATA_ROOT / "nih/train-small.csv"



"""








"""
# 1. Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# 2. Clone repo
!git clone https://github.com/you/project.git
%cd project

# 3. Install deps
!pip install -r requirements.txt

# 4. Run training
!python scripts/train.py --config configs/wide_deep.yaml


-- path to proj.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


-- download model
from google.colab import files
files.download("model.pt")



"""



"""
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/you/my_project.git
%cd my_project
!pip install -r requirements.txt

import os
os.environ["DATA_ROOT"] = "/content/drive/MyDrive/data/my_project"

!python scripts/train.py --config configs/wide_deep.yaml
"""