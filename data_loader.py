import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(data_path, image_size=(96, 96)):
    image_array = []
    label_array = []
    files = os.listdir(data_path)
    
    for i, folder in enumerate(files):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for image_file in tqdm(os.listdir(folder_path)):
                try:
                    img_path = os.path.join(folder_path, image_file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, image_size)
                    image_array.append(img)
                    label_array.append(i)
                except:
                    pass
                    
    image_array = np.array(image_array) / 255.0
    label_array = np.array(label_array)
    return train_test_split(image_array, label_array, test_size=0.15)

# Usage
# X_train, X_test, Y_train, Y_test = load_data("/content/drive/MyDrive/Face_recognition/dataset")
