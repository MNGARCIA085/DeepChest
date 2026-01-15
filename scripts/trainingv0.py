from deep_chest.preprocessors.base import DataModule
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense


from deep_chest.core.paths import get_data_root


from deep_chest.losses.weighted_bce import weighted_bce




DATA_ROOT = get_data_root()

IMAGE_DIR = DATA_ROOT/"images-small/" # then to config!!!!
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







from deep_chest.models.simple_cnn import SimpleCNN


from deep_chest.training.base import Trainer


from deep_chest.training.callbacks import early_stopping


import tensorflow as tf


def main():

    # load data and prepare generators
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
    val_gen = data.val_generator()

    print(train_gen, val_gen)


    # load model; be better later with input_shape and num_classes
    model = SimpleCNN(input_shape=(320, 320, 3), num_classes=14)
    print(model)

    _, _, pos_weights = data.compute_pos_weights_from_generator(train_gen)
    print(pos_weights)



    # 
    callbacks = []
    callbacks.append(
    early_stopping(
            monitor="val_loss",    #"val_auroc",
            patience=3,
        )
    )

    
    # train model (here, without transfer learning bc the model is not applicable for that)
    trainer = Trainer(model=model)

    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=weighted_bce(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, name="auroc"),
            tf.keras.metrics.AUC(curve="PR", multi_label=True, name="auprc"),
        ],
    )

    trainer.fit(train_gen, val_gen, 1, 100, 25, callbacks)


    print(trainer.model)


    # then preds and eval....
    


    
    

    """
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

    




if __name__==main():
    main()
