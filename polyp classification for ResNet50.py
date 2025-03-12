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
    roc_curve, auc, confusion_matrix)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# Build multi-class classification model on top of the pretrained ResNet50
num_classes = 7  # Change this based on the number of your classes
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')  # Multi-class classification output
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

# Load training set
train_annotations = pd.read_csv(train_annotation_file)
train_images = []
train_labels = []

for index, row in train_annotations.iterrows():
    image_name = row['image_name']
    image_path = os.path.join(train_image_folder, image_name)
    processed_image = preprocess_image_resnet(image_path)

    if processed_image is not None:
        train_images.append(processed_image)
        label = row['polyp_type']  # Assuming 'label' is the column containing class information
        train_labels.append(label)
    else:
        print(f"Error loading image: {image_name}")

# Load normal training set
normal_train_images = []
normal_train_labels = []

for image_name in os.listdir(normal_train_folder):
    image_path = os.path.join(normal_train_folder, image_name)
    processed_image = preprocess_image_resnet(image_path)

    if processed_image is not None:
        normal_train_images.append(processed_image)
        label = '0'  # Assuming '0' as the class label for normal images
        normal_train_labels.append(label)
    else:
        print(f"Error loading image: {image_name}")

# Combine normal and polyp training images and labels
train_images.extend(normal_train_images)
train_labels.extend(normal_train_labels)

# Convert lists to numpy arrays
X_train = np.vstack(train_images)
y_train = np.array(train_labels)

# One-hot encode the labels for multi-class classification
y_train = np.eye(num_classes)[y_train.astype(int)]

# Shuffle data to randomize order
X_train, y_train = shuffle(X_train, y_train, random_state=42)

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

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
        label = row['polyp_class']  # Assuming 'label' is the column containing class information
        test_labels.append(label)
    else:
        print(f"Error loading image: {image_name}")

# Load normal testing set
normal_test_images = []
normal_test_labels = []

for image_name in os.listdir(normal_test_folder):
    image_path = os.path.join(normal_test_folder, image_name)
    processed_image = preprocess_image_resnet(image_path)

    if processed_image is not None:
        normal_test_images.append(processed_image)
        label = '0'  # Assuming 'normal' as the class label for normal images
        normal_test_labels.append(label)
    else:
        print(f"Error loading image: {image_name}")

# Combine normal and polyp testing images and labels
test_images.extend(normal_test_images)
test_labels.extend(normal_test_labels)

# Convert lists to numpy arrays
X_test = np.vstack(test_images)
y_test = np.array(test_labels)

# One-hot encode the labels for multi-class classification
y_test = np.eye(num_classes)[y_test.astype(int)]

# Shuffle data to randomize order
X_test, y_test = shuffle(X_test, y_test, random_state=42)

y_pred = model.predict(X_test)

# Convert predicted labels to one-hot encoded format
y_pred_class = np.argmax(y_pred, axis=1)
y_pred_onehot = np.eye(num_classes)[y_pred_class.astype(int)]

# Calculate metrics for each class
class_metrics = {}
for class_label in range(num_classes):  # assuming class labels are integers from 0 to num_classes-1
    class_indices = np.where(y_test[:, class_label] == 1)[0]
    class_y_true = np.zeros_like(y_test[:, class_label])
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


# Define custom class labels
class_labels = ["Normal", "Adenocarcinoma", "Adenomatous-Tubular Polyp", "Adenomatous-Tubulovillous Polyp",
                "Adenomatous-Villous Polyp", "Hyperplastic Polyp", "Inflammatory Polyp"]

# Calculate overall confusion matrix
overall_conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

plt.rcParams.update({
    'font.size': 16,             # Increase font size
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'figure.titlesize': 14
})
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Reds', annot_kws={'size': 20})
plt.title('ResNet50')
plt.xticks(ticks=[i + 0.7 for i in range(len(class_labels))], labels=class_labels, rotation=90, ha='right')
plt.yticks(ticks=[i + 0.5 for i in range(len(class_labels))], labels=class_labels, rotation=0, va='center')

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save confusion matrix as PNG
plt.savefig('confusion_matrix.png', format='png', dpi=400)

# Save confusion matrix as JPEG
plt.savefig('confusion_matrix.jpeg', format='jpeg',dpi=400)

# Save confusion matrix as PDF
plt.savefig('confusion_matrix.pdf', format='pdf',dpi=400)

plt.show()
