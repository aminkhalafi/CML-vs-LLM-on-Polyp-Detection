import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

# Read the data from the Excel file
df = pd.read_excel('D:\\research\\colonoscopy\\ai_colon\\CLIP_BASE_experiment_1_final.xlsx')

# Calculate confusion matrix
actual = df['polyp_type']
predicted = df['bio_clip_classify']
conf_mat = confusion_matrix(actual, predicted, labels=[0, 1, 2, 3, 4, 5, 6])

# Remove the third row (index 2)
conf_mat = conf_mat[:7, :]

# Set font properties globally
plt.rcParams.update({
    'font.size': 16,             # Increase font size
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'figure.titlesize': 14
})

# Draw confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds', annot_kws={'size': 20})  # Increase annotation font size to 20

# Add column and row labels
column_labels = ['Normal', 'Adenocarcinoma', 'Adenomatous-Tubular', 'Adenomatous-Tubulovillous', 'Adenomatous-Villous', 'Hyperplastic', 'Inflammatory']
row_labels = ['Normal', 'Adenocarcinoma', 'Adenomatous-Tubular', 'Adenomatous-Tubulovillous', 'Adenomatous-Villous', 'Hyperplastic', 'Inflammatory']
plt.xticks(ticks=[i + 0.5 for i in range(len(column_labels))], labels=column_labels, rotation=90, ha='right')
plt.yticks(ticks=[i + 0.5 for i in range(len(row_labels))], labels=row_labels, rotation=0, va='center')
# Add labels to the axes
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add title
plt.title('CLIP')
# Adjust margins to prevent label cropping
plt.tight_layout()

# Save the confusion matrix in different formats
plt.savefig('clip_class_confusion_matrix_400dpi.pdf', dpi=400)
plt.savefig('clip_class_confusion_matrix_400dpi.png', dpi=400)
plt.savefig('clip_class_confusion_matrix_400dpi.jpeg', dpi=400)

plt.show()

