from data_loader import load_images_from_folder
from model_builder import build_model, compile_model
from train_model import train_model, split_data
from predict import make_predictions, save

if __name__ == "__main__":
    path = "D:/FutureExpertData/FaceRecognition/EfficientNet_/original"
    X,y,label = load_images_from_folder(path)

    model = build_model()
    model = compile_model(model)
    X_train, X_test, y_train, y_test = split_data(X,y)
    model, history = train_model(model, X_train, X_test, y_train, y_test)
    make_predictions(model, X_test, label)
    print(y_test[:20])
    save(model)
    
    
    
 