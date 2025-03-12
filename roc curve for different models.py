import matplotlib.pyplot as plt

# Example data: replace these with your actual tpr, fpr, auc values for different models
models = {
    'DecisionTreeClassifier': {'fpr': [0.0, 0.1, 0.2, 0.3], 'tpr': [0.0, 0.4, 0.7, 1.0], 'auc': 0.85},
    'RandomForestClassifier': {'fpr': [0.0, 0.2, 0.4, 0.6], 'tpr': [0.0, 0.5, 0.8, 1.0], 'auc': 0.75},
    'SVC': {'fpr': [0.0, 0.3, 0.6, 0.9], 'tpr': [0.0, 0.6, 0.9, 1.0], 'auc': 0.65},
    'LogisticRegression': {'fpr': [0.0, 0.3, 0.6, 0.9], 'tpr': [0.0, 0.6, 0.9, 1.0], 'auc': 0.65},
    'GaussianNB': {'fpr': [0.0, 0.3, 0.6, 0.9], 'tpr': [0.0, 0.6, 0.9, 1.0], 'auc': 0.65},
    'ResNet50': {'fpr': [0.0, 0.3, 0.6, 0.9], 'tpr': [0.0, 0.6, 0.9, 1.0], 'auc': 0.65},
    'chatGPT4V': {'fpr': [0.0, 0.3, 0.6, 0.9], 'tpr': [0.0, 0.6, 0.9, 1.0], 'auc': 0.65},
    'Claude3-opus': {'fpr': [0.0, 0.3, 0.6, 0.9], 'tpr': [0.0, 0.6, 0.9, 1.0], 'auc': 0.65}
}

plt.figure(figsize=(10, 8))

for model_name, data in models.items():
    fpr = data['fpr']
    tpr = data['tpr']
    auc = data['auc']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')

# Plot the random guessing line
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

# Add plot details
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.grid(True)

# Save plot in different formats
plt.savefig('roc_curve.png', format='png')
plt.savefig('roc_curve.pdf', format='pdf')
plt.savefig('roc_curve.jpeg', format='jpeg')

# Show plot
plt.show()
