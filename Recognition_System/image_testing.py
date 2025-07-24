import os
from tqdm import tqdm
from  Data_processing.data_loader import load_images_from_folder, load_image
from scipy.spatial.distance import euclidean
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

"""
This line needs to be executed to active the packages

$env:PYTHONPATH="$env:PYTHONPATH;link_to_current directory "  
"""
# Load TFLite model and allocate tensors
model = load_model("./Models/model_augmented.keras")



# Load the image
image_path1 = "D:\FutureExpertData\FaceRecognition\DeepFaceProject\dataset\Dominique_de_Villepin\Dominique_de_Villepin_0002.jpg"
image_path2 = "D:\FutureExpertData\FaceRecognition\CNN_project\people\Vojislav_Kostunica.jpg"

#Load folder
folder_path = "./people"

'''
# Load and format the image
def format_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (96, 96))  # Resize to match model's expected input
    image = image / 255.0  # Normalize the pixel values
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    return input_data
'''
# Prepare input data for each image
#input_data1 = format_image(image_path1)
#input_data2 = format_image(image_path2)








path = "D:/FutureExpertData/FaceRecognition/EfficientNet_/augmented_dataset"

X, y, _ = load_images_from_folder(path)     
def compare_image(image_path1, image_path2):
    # Load the images from the file paths
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)
    #Predict them
    prediction1 = model.predict(img1).flatten()
    prediction2 = model.predict(img2).flatten()
    # Compare the predictions
    distance = euclidean(prediction1, prediction2)
    threshold = 0.8  # Set an appropriate threshold based on testing
    if distance < threshold:
        return "They are identical"
    else:
        return "They are different"

print(compare_image(image_path1, image_path2))


