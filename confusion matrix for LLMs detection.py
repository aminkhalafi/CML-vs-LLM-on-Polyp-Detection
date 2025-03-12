import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

# Read the data from the Excel file
df = pd.read_excel('D:\\research\\colonoscopy\\ai_colon\\gemini_ex0.xlsx')

# Calculate confusion matrix
actual = df['polypdet']
predicted = df['gemini detection best']
conf_mat = confusion_matrix(actual, predicted, labels=[0, 1, 7])

# Remove the third row (index 2)
conf_mat = conf_mat[:2, :]

# Set font properties globally
plt.rcParams.update({
    'font.size': 25,             # Increase font size
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'figure.titlesize': 25
})

# Draw confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds', annot_kws={'size': 35})
# Add column and row labels
column_labels = ['No Polyp', 'Polyp', 'No Answer']
row_labels = ['No Polyp', 'Polyp']
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=column_labels, rotation=0, ha='center')
plt.yticks(ticks=[0.5, 1.5], labels=row_labels, rotation=90, va='center')
# Add labels to the axes
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add title
plt.title('Gemini 1.5 Pro-best prompt')

# Save the confusion matrix in different formats with specific DPI
plt.savefig('GEMINI_best_confusion_matrix_400dpi.pdf', dpi=400)
plt.savefig('GEMINI_best_confusion_matrix_400dpi.png', dpi=400)
plt.savefig('GEMINI_best_confusion_matrix_400dpi.jpeg', dpi=400)

plt.show()

# Calculate the F1 score
# Removing entries with 'No Answer' (label 7) for F1 score calculation
actual_filtered = actual[actual != 7]
predicted_filtered = predicted[actual != 7]

f1 = f1_score(actual_filtered, predicted_filtered , average='weighted')
print(f'F1 Score: {f1:.4f}')
