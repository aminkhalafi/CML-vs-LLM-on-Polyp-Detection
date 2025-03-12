import os
import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, auc, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_image(image):
    # Check if the image is loaded successfully
    if image is None:
        print("Error: Image not loaded successfully.")
        return None
    
    # Resize image to a fixed size
    resized_image = cv2.resize(image, (300, 300))  # Assuming the images are color images
    
    # Check if the resized image has the expected size
    if resized_image.shape != (300, 300, 3):  # Assuming the images are color images
        print("Error: Resized image has unexpected size.")
        return None
    return resized_image

def load_data(train_annotation_file, train_image_folder, test_annotation_file, test_image_folder, normal_train_folder, normal_test_folder):
    # Load training set
    train_annotations = pd.read_csv(train_annotation_file)
    train_images = []
    train_labels = []

    for index, row in train_annotations.iterrows():
        image_name = row['image_name']
        image_path = os.path.join(train_image_folder, image_name)
        image = cv2.imread(image_path)
        processed_image = preprocess_image(image)
        
        if processed_image is not None:
            train_images.append(processed_image)
            train_labels.append(1)  # 1 for polyp
        else:
            print(f"Error loading image: {image_name}")

    # Load testing set
    test_annotations = pd.read_csv(test_annotation_file)
    test_images = []
    test_labels = []

    for index, row in test_annotations.iterrows():
        image_name = row['image_name']
        image_path = os.path.join(test_image_folder, image_name)
        image = cv2.imread(image_path)
        processed_image = preprocess_image(image)
        
        if processed_image is not None:
            test_images.append(processed_image)
            test_labels.append(1)  # 1 for polyp
        else:
            print(f"Error loading image: {image_name}")



    normal_train_image_list = os.listdir(normal_train_folder)

    for normal_image_name in normal_train_image_list:
        normal_image_path = os.path.join(normal_train_folder, normal_image_name)
        normal_image = cv2.imread(normal_image_path)
        processed_normal_image = preprocess_image(normal_image)
        
        if processed_normal_image is not None:
            train_images.append(processed_normal_image)
            train_labels.append(0)  # 0 for normal
        else:
            print(f"Error loading image: {normal_image_name}")


    normal_test_image_list = os.listdir(normal_test_folder)

    for normal_image_name in normal_test_image_list:
        normal_image_path = os.path.join(normal_test_folder, normal_image_name)
        normal_image = cv2.imread(normal_image_path)
        processed_normal_image = preprocess_image(normal_image)
        
        if processed_normal_image is not None:
            test_images.append(processed_normal_image)
            test_labels.append(0)  # 0 for normal
        else:
            print(f"Error loading image: {normal_image_name}")

    return train_images, train_labels, test_images, test_labels

# Paths to data
train_annotation_file = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\keys\\all_keys_labmod_aug.csv'
train_image_folder = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\all'
test_annotation_file = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\new_test_set_csv.csv'
test_image_folder = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\polyps'
normal_train_folder = 'D:\\research\\colonoscopy\\ai_colon\\normal_train_set'
normal_test_folder = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\normal'

# Load training and testing sets
train_images, train_labels, test_images, test_labels = load_data(train_annotation_file, train_image_folder, test_annotation_file, test_image_folder, normal_train_folder, normal_test_folder)

# Shuffle data to randomize order
train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
test_images, test_labels = shuffle(test_images, test_labels, random_state=42)

# Convert lists to numpy arrays
X_train = np.array(train_images)
y_train = np.array(train_labels)
X_test = np.array(test_images)
y_test = np.array(test_labels)

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

# Define classifiers
classifiers = [
    DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=2, min_samples_split=2),
    RandomForestClassifier(n_estimators=200, min_samples_leaf=1, min_samples_split=10, random_state=42),
    SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    LogisticRegression(C=0.1, max_iter=100, solver='sag', random_state=42),
    GaussianNB()
]


# Directory for saving ROC curves
save_dir = 'D:\\research\\colonoscopy\\ai_colon\\result'

# Initialize an Excel writer object
excel_writer = pd.ExcelWriter('class_metrics_per_cml_model_detection.xlsx')

# Initialize a dictionary to store confusion matrices for each model
overall_confusion_matrix_dict = {}
# Iterate over classifiers
for model in classifiers:
    # Create a dictionary to store results for each model
    model_result = {'Model': type(model).__name__}
    
    # Train the model
    model.fit(X_train_flatten, y_train)
    
    # Make predictions
    y_pred = model.predict_proba(X_test_flatten)
    
    # Convert probabilities to predicted labels
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix for this model
    overall_confusion_matrix_dict[model] = confusion_matrix(y_test, y_pred_labels)

    # Calculate class-wise metrics
    class_metrics = {}
    for class_label in np.unique(y_test):
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

    # Convert class-wise metrics to DataFrame
    class_metrics_df = pd.DataFrame(class_metrics)
    # Transpose the DataFrame to have classes in rows and metrics in columns
    class_metrics_df = class_metrics_df.T

    class_metrics_df.reset_index(inplace=True)
    class_metrics_df.rename(columns={'index': 'Class'}, inplace=True)

    # Save class-wise metrics for this model to a separate sheet
    class_metrics_df.to_excel(excel_writer, sheet_name=type(model).__name__, index=False)
# Save the Excel file
excel_writer._save()

   
# Plot the confusion matrix for each model
for model in classifiers:
    # Set font properties globally
    plt.rcParams.update({
        'font.size': 18,             # Increase font size
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'figure.titlesize': 20
    })

    # Draw confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(overall_confusion_matrix_dict[model], annot=True, fmt='d', cmap='Blues', annot_kws={'size': 20})

    tick_marks = np.arange(2)
    plt.xticks(tick_marks + 0.5, ['Normal', 'Polyp'], rotation=0, ha='center')  # Adding 0.5 to center the labels
    plt.yticks(tick_marks + 0.5, ['Normal', 'Polyp'], rotation=0, va='center')  # Adding 0.5 to center the labels

    plt.tight_layout()

    # Save the plot in different formats
    save_path = os.path.join(save_dir, f'{type(model).__name__}_confusion_matrix')
    plt.savefig(f'{save_path}.png', dpi=400)
    plt.savefig(f'{save_path}.pdf', dpi=400)
    plt.savefig(f'{save_path}.jpeg', dpi=400)
    plt.show()
