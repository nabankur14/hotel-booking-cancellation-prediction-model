import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Perform initial data cleaning and feature engineering.
    """
    data = df.copy()
    
    # Drop Booking_ID as it's an identifier
    if 'Booking_ID' in data.columns:
        data = data.drop('Booking_ID', axis=1)
    
    # Map booking_status to binary
    if data['booking_status'].dtype == 'object':
        data['booking_status'] = data['booking_status'].map({'Not_Canceled': 0, 'Canceled': 1})
        data['booking_status'] = data['booking_status'].astype('int64')
        
    return data

def split_and_scale_data(data, target_col='booking_status', test_size=0.3, random_state=1):
    """
    Split data into training and testing sets, and perform scaling.
    """
    X = data.drop(target_col, axis=1)
    Y = data[target_col]
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )
    
    # Reset index
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True) # Good practice to reset y_test as well
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler
    }
