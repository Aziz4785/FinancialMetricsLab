

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


"""
to run : py -m metrics.Metric5.ml.RF_grid
"""


def optimize_random_forest(df):
    print("length of df : ",len(df))
    # Define feature columns
    
    feature_cols = ['marketCap', 'fwdPriceTosale', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']
    
    feature_cols =['evebitda', 'sma_50d', 'marketCap', 'markRevRatio', 'EVRevenues',
       'fwdPriceTosale', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']
    
    feature_cols =['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'markRevRatio',
       'EVRevenues', 'fwdPriceTosale', 'deriv_2m', '1Y_6M_growth',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth']
    
    feature_cols1 = ['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'EV', 'markRevRatio',
       'peg', 'netDebtToPrice', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'deriv_2m', '4M_return', '1Y_6M_growth', 'sma_50d_to_sma_100d_ratio',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth']
    
    # Prepare X and y
    X = df[feature_cols1]
    y = df['to_buy']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30,40],
        'min_samples_split': [2, 5, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4],
        'max_features': ['sqrt', 'log2',None]
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
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
        'feature': feature_cols1,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return grid_search.best_estimator_



df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = pd.get_dummies(df, columns=['sector'])
best_model = optimize_random_forest(df)
print(best_model)

"""
{'max_depth': 10, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
{'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
"""