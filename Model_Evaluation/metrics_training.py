from sklearn.metrics import classification_report, precision_recall_curve, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
import os
from tqdm import tqdm
from Data_processing.data_loader import load_images_from_folder
from scipy.spatial.distance import euclidean
import tensorflow as tf
import cv2
import numpy as np
from Model_Development.train_model import split_data
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import time

# Load TFLite model and allocate tensors
#   For orginal data
#model = load_model("./model/model_original.keras")
#   For augmented data
model = load_model("./Models/model_augmented.keras")

# Load data
#   Path to the original dataset
#path = "D:/FutureExpertData/FaceRecognition/EfficientNet_/original_dataset"
#   Path to the augmented dataset
path = "D:/FutureExpertData/FaceRecognition/EfficientNet_/augmented_dataset"

X, y, _ = load_images_from_folder(path)
X_train, X_test, Y_train, Y_test = split_data(X, y)

# Calculate class imbalance ratio (CIR)
class_labels = np.argmax(Y_train, axis=1)
class_counts = Counter(class_labels)
majority_class_count = max(class_counts.values())
minority_class_count = min(class_counts.values())
class_imbalance_ratio = majority_class_count / minority_class_count
print(f"Class Imbalance Ratio: {class_imbalance_ratio}")

# Measure inference speed
start_time = time.time()
y_pred_probs = model.predict(X_train)  # Probabilities
end_time = time.time()
speed = (end_time - start_time) / len(X_train)  # Average time per sample
print(f"Inference Speed: {speed:.6f} seconds/sample")

# Convert predictions and labels to class indices
y_pred = y_pred_probs.argmax(axis=1)
y_true = Y_train.argmax(axis=1)

# For multi-class classification, calculate precision-recall for each class
n_classes = y_pred_probs.shape[1]  # Number of classes
precision_curve = []
recall_curve = []
auc_pr_scores = []

# Compute Precision-Recall curve for each class
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_pred_probs[:, i])
    precision_curve.append(precision)
    recall_curve.append(recall)

    # Calculate AUC for the class
    auc_pr = roc_auc_score((y_true == i).astype(int), y_pred_probs[:, i])
    auc_pr_scores.append(auc_pr)

# Calculate macro-average AUC-PR
average_auc_pr = np.mean(auc_pr_scores)

# Print results
print("AUC-PR scores for each class:", auc_pr_scores)
print("Average AUC-PR (Macro-Average):", average_auc_pr)


# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_true, y_pred, average='macro')
print("Precision:", precision)

# Recall
recall = recall_score(y_true, y_pred, average='macro')
print("Recall:", recall)

# F1-Score
f1 = f1_score(y_true, y_pred, average='macro')
print("F1-Score:", f1)

# Binary Cross-Entropy Loss
binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
loss = binary_crossentropy(Y_train, y_pred_probs).numpy()
print("Binary Cross-Entropy Loss:", loss)

# Hit Rate (HR)
hit_rate = sum(y_true == y_pred) / len(y_true)  # Equivalent to accuracy for single-label classification
print("Hit Rate (HR):", hit_rate)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()





