import pandas as pd
import numpy as np
import statsmodels.api as SM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression_statsmodels(X_train, y_train):
    """
    Train Logistic Regression using Statsmodels (for detailed summary).
    """
    X_train_with_intercept = SM.add_constant(X_train)
    model = SM.Logit(y_train, X_train_with_intercept).fit()
    return model

def train_naive_bayes(X_train, y_train):
    """
    Train Gaussian Naive Bayes model.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, k=5):
    """
    Train K-Nearest Neighbors model.
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train Decision Tree Classifier.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def tune_decision_tree(X_train, y_train, random_state=42):
    """
    Tune Decision Tree using GridSearchCV.
    """
    dt_model_tuned = DecisionTreeClassifier(random_state=random_state)

    parameters = {
        "max_depth": np.arange(5, 13, 2),
        "max_leaf_nodes": [10, 20, 40, 50, 75, 100],
        "min_samples_split": [2, 5, 7, 10, 20, 30],
        "class_weight": ['balanced', None]
    }

    grid_obj = GridSearchCV(dt_model_tuned, parameters, scoring='recall', cv=5)
    grid_obj = grid_obj.fit(X_train, y_train)
    
    best_model = grid_obj.best_estimator_
    best_model.fit(X_train, y_train)
    
    return best_model
