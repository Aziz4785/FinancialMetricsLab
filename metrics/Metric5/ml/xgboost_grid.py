

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

"""
to run : py -m metrics.Metric5.ml.xgboost_grid
"""


def optimize_xgboost(df):
    # Define feature columns
    feature_cols = ['RnD_expenses', 'eps_growth', 'deriv_min8M',
       'sma_100d_to_sma_200d_ratio']

    feature_cols2 =['evebitda', 'RnD_expenses', 'marketCap', 'eps_growth', 'peg',
       'EVRevenues', 'fwdPriceTosale', 'var_sma10D_100D', 'deriv_2m',
       'deriv_max4M', 'deriv_min8M', '1Y_6M_growth',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score']
    
    feature_cols2 =['evebitda', 'RnD_expenses', 'marketCap', 'eps_growth', 'peg',
       'fwdPriceTosale', 'var_sma10D_100D', 'deriv_2m', 'deriv_min8M',
       '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']   
    
    feature_cols2 =['RnD_expenses', 'eps_growth', 'peg', 'var_sma10D_100D', 'deriv_min8M',
       '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']  
     
    # Prepare X and y
    X = df[feature_cols]
    y = df['to_buy']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
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
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = pd.get_dummies(df, columns=['sector'])
best_model = optimize_xgboost(df)

"""
{'colsample_bytree': 0.9, 'gamma': 0.2, 'learning_rate': 0.3, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 300, 'subsample': 0.8}
{'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 300, 'subsample': 0.9}
{'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 7, 'min_child_weight': 1, 'n_estimators': 300, 'subsample': 0.7}
{'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 1, 'n_estimators': 300, 'subsample': 0.7}
"""