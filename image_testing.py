import os
from tqdm import tqdm
from data_loader import load_data
from scipy.spatial.distance import euclidean
import tensorflow as tf
import cv2
import numpy as np

# Load and format the image
def format_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (96, 96))  # Resize to match model's expected input
    image = image / 255.0  # Normalize the pixel values
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    return input_data

# Prepare input data for each image
#input_data1 = format_image(image_path1)
#input_data2 = format_image(image_path2)

# Extract image embedding
def get_embedding(input_data):
    # Run inference for the first image
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index']).flatten()
    return embedding



#Create a database with image name and its path
def create_database(folder_path):
    # Initialize an empty dictionary for the database
    database = {}
    
    # Iterate over each file in the specified folder
    for i, image_file in enumerate(tqdm(os.listdir(folder_path), desc="Creating Database")):
        img_path = os.path.join(folder_path, image_file)
        
        # Check if the file is an image (optional, but good practice)
        if os.path.isfile(img_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            database[image_file] = img_path  # Store the image path in the dictionary
    
    return database
   
#Verify whether the data base contains the image
def check_image_in_database(image_path, database):
    #Format image1
    input_data1 = format_image(image_path)
    embedding1 = get_embedding(input_data1)
    id_match = ""
    for key, value in database.items():
        print(value)
        #Format image
        input_data2 = format_image(value)
        # Run inference for the second image
        embedding2 =  get_embedding(input_data2)

        # Calculate the distance between the two embeddings
        distance = euclidean(embedding1, embedding2)
        threshold = 0.8  # Set an appropriate threshold based on testing

        if distance < threshold:
            id_match = key
            break
    if id_match!="":
        return f"The person in the image is {id_match}."
    else:
        return "Person not found in the database."
    
#Compare two images
def compare_images(image_path1, image_path2):
    #Format image1
    input_data1 = format_image(image_path1)
    embedding1 = get_embedding(input_data1)
    #Format image2
    input_data2 = format_image(image_path2)
    embedding2 = get_embedding(input_data2)
    distance = euclidean(embedding1, embedding2)
    threshold = 0.5 # Set an appropriate threshold based on testing
    if distance < threshold:
        return 'These images show the same person'
    else:
        return 'These images show two different people'



# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#Load folder
folder_path = "./people"
  
#Check image in database
image_path = "./dataset/Li_Zhaoxing/Li_Zhaoxing_0002_gaussian_noise.png"
folder_path = "./people"
database = create_database(folder_path)
print(check_image_in_database(image_path, database))

#Compare two images
image_path1 = ".\dataset\Dominique_de_Villepin\Dominique_de_Villepin_0002.jpg"
image_path2 = ".\people\Kim_Dae-jung.jpg"
print(compare_images(image_path1, image_path2))