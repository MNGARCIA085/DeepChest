

class Preprocessor():
	


	def make_image_generator(preprocess_fn=None):
	    if preprocess_fn is None:
	        return ImageDataGenerator(rescale=1./255)
	    return ImageDataGenerator(preprocessing_function=preprocess_fn)


	def get_generator(
	    df,
	    image_dir,
	    x_col,
	    y_cols,
	    preprocess_fn=None,
	    shuffle=True, # only True for train!!!!!
	    batch_size=8,
	    seed=1,
	    target_size=(320, 320),
	):
	    image_generator = make_image_generator(preprocess_fn)

	    return image_generator.flow_from_dataframe(
	        dataframe=df,
	        directory=image_dir,
	        x_col=x_col,
	        y_col=y_cols,
	        class_mode="raw",
	        batch_size=batch_size,
	        shuffle=shuffle,
	        seed=seed,
	        target_size=target_size,
	    )




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