import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Example values for TP, FP, TN, FN
tp = 206
fp = 31
tn = 165
fn = 8

# Create the confusion matrix
confusion_matrix = np.array([[tn, fp],
                             [fn, tp]])

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
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Reds', annot_kws={'size': 35})

tick_marks = np.arange(2)
plt.xticks(tick_marks + 0.5, ['No Polyp', 'Polyp'], rotation=0, ha='center')  # Adding 0.5 to center the labels
plt.yticks(tick_marks + 0.5, ['No Polyp', 'Polyp'], rotation=90, va='center')  # Adding 0.5 to center the labels

# Add labels to the axes
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add title
plt.title('ResNet-50')

plt.tight_layout()

# Save the plot in different formats
plt.savefig('res50_detection.png', dpi=400)
plt.savefig('res50_detection.pdf', dpi=400)
plt.savefig('res50_detection.jpeg', dpi=400)
plt.show()
