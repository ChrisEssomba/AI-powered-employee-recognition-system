# Import necessary libraries
import numpy as np  # For numerical operations
import tensorflow as tf  # For building and training the deep learning model
from tensorflow.keras import layers, models  # For defining the neural network architecture
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import cv2  # For image processing
import os  # For handling file system operations

# Function to split data into training and testing sets
def split_data(X, y):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
    X (array-like): Feature data (e.g., images).
    y (array-like): Labels corresponding to the feature data.
    
    Returns:
    tuple: Four arrays - X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=0.15, random_state=42)

# Function to train the model and evaluate its performance
def train_model(model, X_train, X_test, y_train, y_test):
    """
    Trains the provided model on the given training data and evaluates its performance on test data.
    
    Parameters:
    model (tf.keras.Model): The neural network model to train.
    X_train (array-like): Training feature data.
    X_test (array-like): Testing feature data.
    y_train (array-like): Training labels.
    y_test (array-like): Testing labels.
    
    Returns:
    tuple: The trained model and the training history.
    """
    
    # Train the model on the training data and validate on the test data
    history = model.fit(
        X_train, y_train,  # Input training data and labels
        epochs=10,  # Number of training epochs
        batch_size=32,  # Number of samples per training batch
        validation_data=(X_test, y_test)  # Validation data for monitoring performance
    )

    # Evaluate the model's performance on the test data
    print("\nEvaluating the model on test data:")
    loss, acc, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)  # Suppresses detailed output
    print(f"accuracy {acc:.4f}, AUC: {auc:.4f},  precision: {precision:.4f}, recall: {recall:.4f}")  # Print evaluation metrics

    return model, history  # Return the trained model and its training history



'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
import os

def split_data(X,y):
    return train_test_split(X, y, test_size=0.15, random_state=42)

def train_model(model, X_train, X_test, y_train, y_test):
    
    
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    
    print("\nEvaluating the model on test data:")
    loss, acc, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"accuracy {acc:.4f}, AUC: {auc:.4f},  precision: {precision:.4f}, recall: {recall:.4f}")
    

    return model, history
'''