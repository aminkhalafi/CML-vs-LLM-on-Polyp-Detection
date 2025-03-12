import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Read the data from the Excel file
df = pd.read_excel('D:\\research\\colonoscopy\\ai_colon\\gemini - experiment 1.xlsx')

# Calculate confusion matrix
actual = df['polypdet']
predicted = df['gemini detection best']
conf_mat = confusion_matrix(actual, predicted, labels=[0, 1, 7])


# Calculate ROC curve, AUC, TPR, and FPR for each class
roc_auc_data = {'Class': ['No Polyp', 'Polyp'], 'AUC': [], 'TPR': [], 'FPR': []}
for i in range(2):
    fpr, tpr, _ = roc_curve((actual == i).astype(int), (predicted == i).astype(int))
    roc_auc_data['AUC'].append(auc(fpr, tpr))
    roc_auc_data['TPR'].append(tpr)
    roc_auc_data['FPR'].append(fpr)

# Store the results in a DataFrame
roc_auc_df = pd.DataFrame(roc_auc_data)

# Write the DataFrame to an Excel file
roc_auc_df.to_excel('gemini_det_roc_auc_results.xlsx', index=False)
