from data_loader import load_data
from model_builder import build_model, compile_model
from train_model import train_model
from predict import save_as_tflite, make_predictions

if __name__ == "__main__":
    path = "./dataset"
    X_train, X_test, Y_train, Y_test = load_data(path)

    model = build_model()
    model = compile_model(model)
    model, history = train_model(model, X_train, Y_train, X_test, Y_test)

    save_as_tflite(model, "model.tflite")
    make_predictions(model, X_test)
