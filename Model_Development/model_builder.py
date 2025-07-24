
# Import necessary libraries
import numpy as np  # For numerical computations
import tensorflow as tf  # Core library for deep learning
from tensorflow.keras import layers, models  # For defining and building neural networks
from sklearn.model_selection import train_test_split  # For splitting datasets
import cv2  # For image processing
import os  # For file and directory management

# Function to build the model architecture
def build_model(input_shape=(96, 96, 3)):
    """
    Builds a convolutional neural network (CNN) for image classification.
    
    Parameters:
    input_shape (tuple): Shape of the input images, default is (96, 96, 3) for RGB images.
    
    Returns:
    model (tf.keras.Model): Compiled CNN model.
    """
    # Define a sequential model
    model = models.Sequential([
        # First convolutional layer followed by max pooling
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer followed by max pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer followed by max pooling
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the feature maps into a single vector
        layers.Flatten(),
        
        # Fully connected (dense) layer with 128 neurons and ReLU activation
        layers.Dense(128, activation='relu'),
        
        # Optionally, a dropout layer can be added to reduce overfitting
        # layers.Dropout(0.5),
        
        # Final output layer with softmax activation for multi-class classification
        # Number of output neurons corresponds to the number of classes (32 in this case)
        layers.Dense(32, activation='softmax')  
    ])
    return model 

# Function to compile the model
def compile_model(model):
    """
    Compiles the CNN model with the specified optimizer, loss function, and metrics.
    
    Parameters:
    model (tf.keras.Model): The CNN model to compile.
    
    Returns:
    model (tf.keras.Model): Compiled CNN model.
    """
    model.compile(
        optimizer="adam",  # Adam optimizer for adaptive learning rate
        loss="categorical_crossentropy",  # Loss function for multi-class classification
        metrics=[
            "accuracy",  # Standard accuracy metric
            tf.keras.metrics.AUC(name="AUC"),  # Area under the receiver operating characteristic curve
            tf.keras.metrics.Precision(name="Precision"),  # Precision (positive predictive value)
            tf.keras.metrics.Recall(name="Recall")  # Recall (sensitivity)
        ]
    )
    return model

'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
import os

def build_model(input_shape=(96, 96, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        #layers.Dropout(0.5),
        layers.Dense(32, activation='softmax')  # Number of classes
    ])
    return model 

def compile_model(model):
    model.compile(
        optimizer="adam",  # Optimizer for training
        loss="categorical_crossentropy",  # Loss function for multi-class classification
        metrics=[
            "accuracy",  # Basic accuracy
            tf.keras.metrics.AUC(name="AUC"),  # Area under the curve metric
            tf.keras.metrics.Precision(name="Precision"),  # Precision metric
            tf.keras.metrics.Recall(name="Recall")  # Recall metric
        ]
    )
    return model

# Usage example:
# model = build_model()
# model = compile_model(model)
# model.summary()
'''