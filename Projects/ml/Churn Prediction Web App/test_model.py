# test_model.py
import pytest
import pandas as pd
from model import split_data, scale_features, build_model

def test_split_data():
    df = pd.DataFrame({
        'feature1': range(100),
        'Exited': [0, 1] * 50
    })
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert 'Exited' not in X_train.columns

def test_scale_features():
    X_train = pd.DataFrame({'feature': [1, 2, 3]})
    X_test = pd.DataFrame({'feature': [4]})
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    assert X_train_scaled.shape == X_train.shape
    assert scaler is not None
