from deep_chest.preprocessors.base import DataModule
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense


from deep_chest.core.paths import get_data_root


DATA_ROOT = get_data_root()

IMAGE_DIR = DATA_ROOT/"images-small/"
TRAIN_PATH = DATA_ROOT/ "train-small.csv"
VALID_PATH = DATA_ROOT/ "valid-small.csv"
TEST_PATH = DATA_ROOT/ "test.csv"


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





def main():
    data = DataModule(
        train_csv=TRAIN_PATH,
        val_csv=VALID_PATH,
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=labels,
        preprocess_fn=None,
    )
    data.load_data()
    train_gen = data.train_generator()

    print(train_gen)


    
    
    data_eff = DataModule(
        train_csv=TRAIN_PATH,
        val_csv=VALID_PATH,
        test_csv=TEST_PATH,
        image_dir=IMAGE_DIR,
        labels=labels,
        preprocess_fn=preprocess_input_effnet,
    )
    data_eff.load_data()
    train_gen_eff = data_eff.train_generator()
    print(train_gen_eff)

    
    """
    data = ChestXrayDataModule(
        ...,
        preprocess_fn=preprocess_input,
    )
    """



if __name__==main():
    main()
