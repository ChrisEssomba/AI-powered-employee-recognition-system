import tensorflow as tf
from tensorflow.keras import Sequential, layers

def build_model(input_shape=(96, 96, 3)):
    model = Sequential()
    pretrained_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape,
                                                            include_top=False,
                                                            weights="imagenet")
    model.add(pretrained_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))  # Single output for regression
    return model

def compile_model(model):
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

# Usage example:
# model = build_model()
# model = compile_model(model)
# model.summary()
