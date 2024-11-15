import tensorflow as tf
from model_builder import build_model, compile_model
from data_loader import load_data

def train_model(model, X_train, Y_train, X_test, Y_test, epochs=300, batch_size=64):
    ckp_path = "./trained_model/model.weights.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                          monitor="val_mae",
                                                          mode="auto",
                                                          save_best_only=True,
                                                          save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                                     monitor="val_mae",
                                                     mode="auto",
                                                     cooldown=0,
                                                     patience=5,
                                                     verbose=1,
                                                     min_lr=1e-6)

    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[model_checkpoint, reduce_lr])
    model.load_weights(ckp_path)
    return model, history
