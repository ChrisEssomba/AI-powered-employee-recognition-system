import tensorflow as tf
from train_model import train_model
from data_loader import load_data

def save_as_tflite(model, output_path="model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

def make_predictions(model, X_test, num_samples=20):
    predictions = model.predict(X_test, batch_size=64)
    print("Predictions:", predictions[:num_samples])

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data("./dataset")
    model = train_model(X_train, Y_train, X_test, Y_test)
    save_as_tflite(model)
    make_predictions(model, X_test)
