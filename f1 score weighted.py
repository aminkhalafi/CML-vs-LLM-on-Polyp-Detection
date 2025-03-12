# Define the data
models = [
    "CML - DT", "CML - RF", "CML - SVM", "CML - LR", "CML - GNB", 
    "CNN - ResNet50", "VLM - GPT4", "VLM - Claude3Opus", "VLM - Gemini 1.5 Pro", "CLIP", "BiomedCLIP"
]

f1_scores = {
    "CML - DT": [5.71, 52.52, 26.66, 0, 21.62, 14.28],
    "CML - RF": [0, 63.56, 0, 0, 7.70, 0],
    "CML - SVM": [45.45, 67.52, 25.00, 0, 31.25, 36.36],
    "CML - LR": [10.00, 56.33, 0, 0, 6.67, 20.00],
    "CML - GNB": [7.70, 9.21, 0, 0, 7.14, 0],
    "CNN - ResNet50": [66.67, 84.77, 54.55, 25.00, 48.98, 71.43],
    "VLM - GPT4": [30.00, 58.13, 0, 0, 0, 0],
    "VLM - Claude3Opus": [18.92, 32.72, 5.71, 3.64, 14.28, 0],
    "VLM - Gemini 1.5 Pro": [0, 9.23, 0, 0, 0, 0],
    "CLIP": [19.04, 0, 0, 0, 0, 3.98],
    "BiomedCLIP": [56.25, 29.29, 16.67, 0, 21.05, 4.28]
}

supports = {
    "CML - DT": [1+16, 73+71, 4+8, 0+7, 4+21, 1+8],
    "CML - RF": [0+17, 109+35, 0+12, 0+7, 1+24, 0+9],
    "CML - SVM": [5+12, 106+38, 2+10, 0+7, 5+20, 2+7],
    "CML - LR": [1+16, 89+55, 0+12, 0+7, 1+24, 1+8],
    "CML - GNB": [1+16, 7+137, 0+12, 0+7, 1+24, 0+9],
    "CNN - ResNet50": [9+8, 128+16, 6+6, 1+6, 12+13, 5+4],
    "VLM - GPT4": [3+11, 59+62, 0+11, 0+6, 0+22, 0+7],
    "VLM - Claude3Opus": [7+7, 36+85, 1+10, 1+5, 9+13, 0+7],
    "VLM - Gemini 1.5 Pro": [0+14, 6+115, 0+11, 0+6, 0+22, 0+7],
    "CLIP": [2+15, 0+142, 0+12, 0+7, 0+25, 8+1],
    "BiomedCLIP": [9+8, 29+113, 4+8, 0+7, 6+19, 3+6]
}

# Calculate weighted F1 score for each model
weighted_f1_scores = {}
for model in models:
    total_support = sum(supports[model])
    weighted_f1 = sum(f1 * support for f1, support in zip(f1_scores[model], supports[model])) / total_support
    weighted_f1_scores[model] = weighted_f1

print(weighted_f1_scores)
