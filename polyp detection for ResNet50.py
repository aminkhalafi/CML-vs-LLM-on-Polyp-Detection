import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, auc, confusion_matrix)
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
import seaborn as sns
import openpyxl
from tensorflow.keras.utils import to_categorical

# Function to preprocess image for ResNet50
def preprocess_image_resnet(image_path):
    img = load_img(image_path, target_size=(300, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Load ResNet50 model
base_model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(300, 300, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Build binary classification model on top of the pretrained ResNet50
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(2, activation='softmax')  # Two classes: polyp and normal
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Paths to data
train_annotation_file = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\keys\\all_keys_labmod_aug.csv'
train_image_folder = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\all'
test_annotation_file = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\new_test_set_csv.csv'
test_image_folder = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\polyps'
normal_train_folder = 'D:\\research\\colonoscopy\\ai_colon\\normal_train_set'
normal_test_folder = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\normal'

# Load training and testing sets
train_annotations = pd.read_csv(train_annotation_file)
train_images = []
train_labels = []

for index, row in train_annotations.iterrows():
    image_name = row['image_name']
    image_path = os.path.join(train_image_folder, image_name)
    processed_image = preprocess_image_resnet(image_path)

    if processed_image is not None:
        train_images.append(processed_image)
        train_labels.append(row['polypdet'])  # Add label for each class
    else:
        print(f"Error loading image: {image_name}")

# Load testing set
test_annotations = pd.read_csv(test_annotation_file)
test_images = []
test_labels = []

for index, row in test_annotations.iterrows():
    image_name = row['image_name']
    image_path = os.path.join(test_image_folder, image_name)
    processed_image = preprocess_image_resnet(image_path)

    if processed_image is not None:
        test_images.append(processed_image)
        test_labels.append(row['polyp_detection'])  # Add label for each class
    else:
        print(f"Error loading image: {image_name}")

# Load normal training set
normal_train_images = os.listdir(normal_train_folder)

for normal_image_name in normal_train_images:
    normal_image_path = os.path.join(normal_train_folder, normal_image_name)
    processed_normal_image = preprocess_image_resnet(normal_image_path)

    if processed_normal_image is not None:
        train_images.append(processed_normal_image)
        train_labels.append(0)  # 2 for normal class
    else:
        print(f"Error loading image: {normal_image_name}")

# Load normal testing set
normal_test_images = os.listdir(normal_test_folder)

for normal_image_name in normal_test_images:
    normal_image_path = os.path.join(normal_test_folder, normal_image_name)
    processed_normal_image = preprocess_image_resnet(normal_image_path)

    if processed_normal_image is not None:
        test_images.append(processed_normal_image)
        test_labels.append(0)  # 2 for normal class
    else:
        print(f"Error loading image: {normal_image_name}")

# Convert lists to numpy arrays
X_train = np.vstack(train_images)
y_train = np.array(train_labels)
X_test = np.vstack(test_images)
y_test = np.array(test_labels)

# Shuffle data to randomize order
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Convert y_train to integers
y_train = y_train.astype(int)
y_train_encoded = to_categorical(y_train, num_classes=2)
y_train_encoded = y_train_encoded.astype(int)

# Convert y_test to integers
y_test = y_test.astype(int)
y_test_encoded = to_categorical(y_test, num_classes=2)
y_test_encoded = y_test_encoded.astype(int)

model.fit(X_train, y_train_encoded, epochs=15, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate metrics for each class
class_metrics = {}
for class_label in range(2):
    class_indices = np.where(y_test == class_label)[0]
    class_y_true = np.zeros_like(y_test)
    class_y_true[class_indices] = 1
    class_y_pred = y_pred[:, class_label]
    class_fpr, class_tpr, _ = roc_curve(class_y_true, class_y_pred)
    class_roc_auc = auc(class_fpr, class_tpr)
    class_conf_matrix = confusion_matrix(class_y_true, class_y_pred.round())
    class_tp = class_conf_matrix[1, 1]
    class_fp = class_conf_matrix[0, 1]
    class_tn = class_conf_matrix[0, 0]
    class_fn = class_conf_matrix[1, 0]
    class_accuracy = accuracy_score(class_y_true, class_y_pred.round())
    class_precision = precision_score(class_y_true, class_y_pred.round())
    class_recall = recall_score(class_y_true, class_y_pred.round())
    class_f1 = f1_score(class_y_true, class_y_pred.round())
    class_metrics[class_label] = {
        'TPR': class_tpr.tolist(),
        'FPR': class_fpr.tolist(),
        'AUC': class_roc_auc,
        'ROC': {'FPR': class_fpr.tolist(), 'TPR': class_tpr.tolist()},
        'Confusion Matrix': class_conf_matrix.tolist(),
        'TP': class_tp,
        'FP': class_fp,
        'TN': class_tn,
        'FN': class_fn,
        'Accuracy': class_accuracy,
        'Precision': class_precision,
        'Recall': class_recall,
        'F1-Score': class_f1
    }

# Convert the class-wise metrics to a DataFrame
class_metrics_df = pd.DataFrame.from_dict(class_metrics, orient='index')

# Save the DataFrame to a CSV file
class_metrics_df.to_csv('class_metrics.csv', index=True)

# Plot confusion matrices and save in different formats
def plot_and_save_confusion_matrix(cm, class_name, output_dir='.', formats=['png', 'pdf', 'jpeg']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Polyp'], yticklabels=['Normal', 'Polyp'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {class_name}')
    for file_format in formats:
        output_path = os.path.join(output_dir, f'confusion_matrix_class_{class_name}.{file_format}')
        plt.savefig(output_path, format=file_format)
    plt.close()

# Directory to save the confusion matrices
output_dir = 'confusion_matrices'
os.makedirs(output_dir, exist_ok=True)

# Plot and save confusion matrices for each class
for class_label, metrics in class_metrics.items():
    cm = np.array(metrics['Confusion Matrix'])
    plot_and_save_confusion_matrix(cm, class_label, output_dir=output_dir)

print("Confusion matrices saved in formats: PNG, PDF, and JPEG.")
