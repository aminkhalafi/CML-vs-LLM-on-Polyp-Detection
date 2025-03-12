import os
import cv2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Define function to load images
def load_images(folder_path, image_names):
    images = []
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))
        images.append(image)
    return np.array(images)

# Paths to datasets
train_df = pd.read_csv('D:\\research\\colonoscopy\\ai_colon\\polyp_images\\keys\\all_keys_labmod_aug.csv')
test_df = pd.read_csv('D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\new_test_set_csv.csv')
train_folder = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\all'
test_folder = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\polyps'
normal_train_folder = 'D:\\research\\colonoscopy\\ai_colon\\normal_train_set'
normal_test_folder = 'D:\\research\\colonoscopy\\ai_colon\\test_set\\new_test_set\\normal'

# Load images
X_train = load_images(train_folder, train_df['image_name'].values)
y_train = train_df['polyp_type'].values

X_test = load_images(test_folder, test_df['image_name'].values)
y_test = test_df['polyp_class'].values

# Load normal images
normal_train_images = []
normal_train_labels = []
for image_name in os.listdir(normal_train_folder):
    image_path = os.path.join(normal_train_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    normal_train_images.append(image)
    normal_train_labels.append(0)  # Label for normal images

normal_test_images = []
normal_test_labels = []
for image_name in os.listdir(normal_test_folder):
    image_path = os.path.join(normal_test_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    normal_test_images.append(image)
    normal_test_labels.append(0)  # Label for normal images

# Combine normal and polyp images
X_train_all = np.vstack([X_train, normal_train_images])
y_train_all = np.concatenate([y_train, normal_train_labels])

X_test_all = np.vstack([X_test, normal_test_images])
y_test_all = np.concatenate([y_test, normal_test_labels])

# Reshape images
X_train_all_flatten = X_train_all.reshape(X_train_all.shape[0], -1)
X_test_all_flatten = X_test_all.reshape(X_test_all.shape[0], -1)

# Define classifiers
classifiers = [
    DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=2, min_samples_split=2),
    RandomForestClassifier(n_estimators=200, min_samples_leaf=1, min_samples_split=10, random_state=42),
    SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    LogisticRegression(C=0.1, max_iter=100, solver='sag', random_state=42),
    GaussianNB()
]

# Initialize data storage
results = []
excel_writer = pd.ExcelWriter('class_metrics_per_cml_model_classify.xlsx')

# Set font properties globally
plt.rcParams.update({
    'font.size': 16,             # Increase font size
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'figure.titlesize': 14
})

# Loop through each classifier
for model in classifiers:
    result_dict = {'Model': type(model).__name__}
    
    # Fit the model
    model.fit(X_train_all_flatten, y_train_all)
    y_pred_all = model.predict(X_test_all_flatten)

    # Evaluation metrics
    accuracy = accuracy_score(y_test_all, y_pred_all)
    precision = precision_score(y_test_all, y_pred_all, average='weighted')
    recall = recall_score(y_test_all, y_pred_all, average='weighted')
    f1 = f1_score(y_test_all, y_pred_all, average='weighted')
    conf_matrix = confusion_matrix(y_test_all, y_pred_all)
    conf_matrix_dict = {'Confusion Matrix': conf_matrix.tolist()}
    
    # Save results to dictionary
    result_dict.update({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': conf_matrix_dict
    })

    # Append results to the list
    results.append(result_dict)

    # Calculate class-wise metrics
    class_metrics = {}
    for class_label in range(len(np.unique(y_test_all))):
        class_indices = np.where(y_test_all == class_label)[0]
        class_y_true = np.zeros_like(y_test_all)
        class_y_true[class_indices] = 1
        class_y_pred = (y_pred_all == class_label).astype(int)

        class_conf_matrix = confusion_matrix(class_y_true, class_y_pred)
        class_tp = class_conf_matrix[1, 1]
        class_fp = class_conf_matrix[0, 1]
        class_tn = class_conf_matrix[0, 0]
        class_fn = class_conf_matrix[1, 0]

        class_accuracy = accuracy_score(class_y_true, class_y_pred)
        class_precision = precision_score(class_y_true, class_y_pred)
        class_recall = recall_score(class_y_true, class_y_pred)
        class_f1 = f1_score(class_y_true, class_y_pred)
        class_fpr, class_tpr, _ = roc_curve(class_y_true, class_y_pred)
        class_roc_auc = auc(class_fpr, class_tpr)

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
    
    # Convert class-wise metrics to DataFrame
    class_metrics_df = pd.DataFrame(class_metrics)
    # Transpose the DataFrame to have classes in rows and metrics in columns
    class_metrics_df = class_metrics_df.T

    class_metrics_df.reset_index(inplace=True)
    class_metrics_df.rename(columns={'index': 'Class'}, inplace=True)

    # Save class-wise metrics for this model to a separate sheet
    class_metrics_df.to_excel(excel_writer, sheet_name=type(model).__name__, index=False)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', annot_kws={'size': 20})

    class_labels = ["Normal", "Adenocarcinoma", "Adenomatous-Tubular Polyp", "Adenomatous-Tubulovillous Polyp",
                    "Adenomatous-Villous Polyp", "Hyperplastic Polyp", "Inflammatory Polyp"]
    plt.xticks(ticks=[i + 0.5 for i in range(len(class_labels))], labels=class_labels, rotation=90, ha='right')
    plt.yticks(ticks=[i + 0.5 for i in range(len(class_labels))], labels=class_labels, rotation=0, va='center')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{type(model).__name__}')
    plt.tight_layout()

    # Save confusion matrix in different formats
    plt.savefig(f'confusion_matrix_{type(model).__name__}.png', dpi=400)
    plt.savefig(f'confusion_matrix_{type(model).__name__}.jpeg', dpi=400)
    plt.savefig(f'confusion_matrix_{type(model).__name__}.pdf', dpi=400)

    plt.show()

# Save results to Excel file
results_df = pd.DataFrame(results)
results_df.to_excel('model_results_polypclass_wo_box.xlsx', index=False)
excel_writer._save()
