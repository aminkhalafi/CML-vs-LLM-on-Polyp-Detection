import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Read the data from the Excel file
df = pd.read_excel('D:\\research\\colonoscopy\\ai_colon\\gemini_ex0.xlsx')

# Initialize the data structure for storing the metrics
classes = ['Normal', 'adenocarcinoma', 'adenomatous-tubular', 'adenomatous-tubulovillous', 'adenomatous-villous', 'hyperplastic', 'inflammatory']
roc_auc_data = {
    'Class': classes,
    'AUC': [], 'TPR': [], 'FPR': [],
    'TP': [], 'FP': [], 'TN': [], 'FN': [],
    'F1 Score': [], 'Accuracy': [], 'Recall': [], 'Precision': []
}

# Calculate metrics for each class
for i, class_name in enumerate(classes):
    actual_class = (df['polyp_type'] == i).astype(int)
    predicted_class = (df['gemini classification simple'] == i).astype(int)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(actual_class, predicted_class)
    roc_auc_data['AUC'].append(auc(fpr, tpr))
    roc_auc_data['TPR'].append(tpr)
    roc_auc_data['FPR'].append(fpr)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(actual_class, predicted_class).ravel()
    roc_auc_data['TP'].append(tp)
    roc_auc_data['FP'].append(fp)
    roc_auc_data['TN'].append(tn)
    roc_auc_data['FN'].append(fn)
    
    # Calculate F1 Score, Accuracy, Recall, and Precision
    roc_auc_data['F1 Score'].append(f1_score(actual_class, predicted_class))
    roc_auc_data['Accuracy'].append(accuracy_score(actual_class, predicted_class))
    roc_auc_data['Recall'].append(recall_score(actual_class, predicted_class))
    roc_auc_data['Precision'].append(precision_score(actual_class, predicted_class))

# Store the results in a DataFrame
roc_auc_df = pd.DataFrame(roc_auc_data)

# Write the DataFrame to an Excel file
roc_auc_df.to_excel('results_geminiex0raw.xlsx', index=False)
