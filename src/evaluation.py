import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

def model_performance_classification(model, predictors, target, threshold=0.5):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    prob_pred = model.predict(predictors)
    
    # Check if model returns probabilities (like for some sklearn models with predict_proba, 
    # but here typically .predict() returns classes for some, but let's stick to the logic 
    # from the original notebook which seemed to act on predictions directly or maybe it assumed prob_pred was probabilities?
    # Wait, looking at original code: 
    # prob_pred = model.predict(predictors)
    # class_pred = [1 if i >= threshold else 0 for i in prob_pred]
    # This implies model.predict returned probabilities or continuous values?
    # Actually for sklearn LogisticRegression .predict returns classes. 
    # For statsmodels Logit .predict returns probabilities.
    # The original notebook used statsmodels for Logit and sklearn for others.
    # I need to handle this.
    
    # If the model is from sklearn, predict returns classes usually.
    # If statsmodels, it returns probabilities.
    
    # Let's try to detect or just trust the input is compatible. 
    # Ideally we should use predict_proba for sklearn if we want to apply threshold.
    
    # However, to be safe and consistent with original code logic:
    try:
        # Check if it looks like probabilities
        if np.any((prob_pred >= 0) & (prob_pred <= 1)) and np.all(prob_pred <= 1.0) and np.all(prob_pred >= 0.0) and len(np.unique(prob_pred)) > 2:
             class_pred = [1 if i >= threshold else 0 for i in prob_pred]
        else:
             class_pred = prob_pred # Assume it's already classes
    except:
        class_pred = prob_pred

    acc = accuracy_score(target, class_pred)  # to compute Accuracy
    recall = recall_score(target, class_pred)  # to compute Recall
    precision = precision_score(target, class_pred)  # to compute Precision
    f1 = f1_score(target, class_pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1": f1},
        index=[0],
    )

    return df_perf

def plot_confusion_matrix(model, predictors, target, threshold=0.5):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    prob_pred = model.predict(predictors)
    
    # Logic to handle probabilities vs classes, same as above
    try:
         if np.any((prob_pred >= 0) & (prob_pred <= 1)) and np.all(prob_pred <= 1.0) and np.all(prob_pred >= 0.0) and len(np.unique(prob_pred)) > 2:
             class_pred = [1 if i >= threshold else 0 for i in prob_pred]
         else:
             class_pred = prob_pred
    except:
        class_pred = prob_pred

    cm = confusion_matrix(target, class_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
