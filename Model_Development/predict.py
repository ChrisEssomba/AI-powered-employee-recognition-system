# Import necessary libraries
import numpy as np  # For numerical operations
import tensorflow as tf  # For building and using machine learning models

# Function to save the model to a file
def save(model):
    """
    Saves the provided model to a file in the Keras format.
    
    Parameters:
    model (tf.keras.Model): The model to save.
    """
    model.save("./model/origin_model.keras")  # Save the model to the specified filepath

# Function to make predictions using the model
def make_predictions(model, X_test, labels, num_samples=20):
    """
    Generates predictions using the provided model on test data.
    
    Parameters:
    model (tf.keras.Model): The trained model to use for predictions.
    X_test (array-like): Test feature data.
    labels (array-like): Labels corresponding to the test data (optional, for context).
    num_samples (int): Number of samples to display predictions for (default is 20).
    """
    
    # Predict on the entire test dataset with a batch size of 64
    predictions = model.predict(X_test, batch_size=64)  
    print("Predictions:", predictions[:num_samples])  # Display raw predictions for the specified number of samples
    
    # Predict only on a subset of the test data (the first `num_samples` samples)
    predictions = model.predict(X_test[:num_samples])  
    
    # Convert the model's output probabilities to numeric class labels
    numeric_predictions = np.argmax(predictions, axis=1)  
    
    # Display the numeric predictions
    print("Numeric Predictions:", numeric_predictions)


'''
import numpy as np
import tensorflow as tf

def save(model):
    model.save("./model/origin_model.keras")


def make_predictions(model, X_test, labels, num_samples=20):
    predictions = model.predict(X_test, batch_size=64)
    print("Predictions:", predictions[:num_samples])
    
   
    predictions = model.predict(X_test[:num_samples])
 
    numeric_predictions = np.argmax(predictions, axis=1)

 

    print("Numeric Predictions:", numeric_predictions)

'''