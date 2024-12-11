

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

"""
to run : py -m metrics.Metric8.ml.SVC_grid
"""

def optimize_svc(df):
    print("length of df : ", len(df))
    # Define feature columns
    test_size_ = 0.2

    feature_cols1 = ['sma_10d_6months_ago', 'marketCap', 'balance_lag_days', 'max_minus_min',
       'min8M_lag', 'Spread4Mby8M', 'markRevRatio', 'peg', 'EVEbitdaRatio',
       'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m',
       'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m', '1y_return',
       'combined_valuation_score']

    # Prepare X and y
    X = df[feature_cols1]
    y = df['to_buy']
    
    print("length of X : ", len(X))
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_)
    print("length of X_train : ", len(X_train))

    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'class_weight': ['balanced', None],
        'degree': [2, 3, 4]  # Only used with poly kernel
    }
    
    # Initialize SVC
    svc = SVC(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='precision',
        verbose=2
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # For SVC, we can get feature importance only for linear kernel
    if best_model.kernel == 'linear':
        feature_importance = pd.DataFrame({
            'feature': feature_cols1,
            'importance': np.abs(best_model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance (only available for linear kernel):")
        print(feature_importance)
    
    return grid_search.best_estimator_

df = pd.read_csv('metrics/Metric8/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = pd.get_dummies(df, columns=['sector'])
best_model = optimize_svc(df)
print(best_model)

#{'C': 1, 'class_weight': None, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}