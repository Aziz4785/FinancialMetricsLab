

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

"""
to run : py -m metrics.Metric8.ml.xgboost_grid
"""
df = pd.read_csv('metrics/Metric8/ml/processed_data.csv')


def optimize_xgboost(df):

    test_size_ = 0.2 #put 0.2
    # Define feature columns
    feature_cols =['month', 'ebitda', 'cashcasheq', 'netDebt', 'curr_est_eps',
       'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M', 'debtToPrice',
       'PS_to_PEG', 'dividend_payout_ratio', 'fwdPriceTosale_diff']

    # Prepare X and y
    X = df[feature_cols]
    y = df['to_buy']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_, random_state=42)
    print("length of xtrain : ",len(X_train))
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7,8],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6,0.7, 0.8, 0.9],
        'colsample_bytree': [0.6,0.7, 0.8, 0.9],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='precision',
        verbose=1
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
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    print("\nTo visualize feature importance, you can use:")
    print("xgb.plot_importance(best_model)")
    
    return grid_search.best_estimator_

# Example usage:
df = df.sample(frac=1).reset_index(drop=True)
df = pd.get_dummies(df, columns=['sector'])
best_model = optimize_xgboost(df)

"""
"""