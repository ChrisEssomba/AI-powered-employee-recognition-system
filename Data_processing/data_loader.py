import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(folder_path, img_size=(96, 96)):
    """
    Load images and their labels from a folder structure where each subfolder represents a class.
    Args:
        folder_path (str): Path to the dataset folder.
        img_size (tuple): Desired image size for resizing.
    Returns:
        images (np.array): Array of processed images.
        labels (np.array): Numeric labels for the images.
        label_map (dict): Mapping of class names to numeric labels.
        reverse_label_map (dict): Mapping of numeric labels back to class names.
    """
    images, labels = [], []
    
    # Create the label map with numeric encoding for class names
    class_names = sorted(os.listdir(folder_path))  # Ensure consistent ordering
    label_map = {label: idx for idx, label in enumerate(class_names)}
    reverse_label_map = {idx: label for label, idx in label_map.items()}  # Reverse mapping

    # Load images and their corresponding labels
    for label in class_names:
        class_folder = os.path.join(folder_path, label)
        for file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values to [0, 1]
                images.append(img)
                labels.append(label_map[label])

    # Convert labels to categorical format
    labels = np.array(labels)
    num_classes = len(label_map)
    categorical_labels = to_categorical(labels, num_classes=num_classes)

    return np.array(images), categorical_labels, label_map


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
            img = cv2.resize(img, (96, 96))
            img = img / 255.0  # Normalize to [0, 1]
    return img
