

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


"""
to run : py -m metrics.Metric9.ml.RF_grid
"""


def optimize_random_forest(df):
    print("length of df : ",len(df))
    # Define feature columns
    test_size_=0.2 #put 0.2

    feature_cols1 =['price', 'sma_50w', 'sma_100d', 'peg', 'EVRevenues',
       'combined_valuation_score']


    # Prepare X and y
    X = df[feature_cols1]
    y = df['to_buy']
    
    print("length of X : ",len(X))
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_, random_state=42)
    print("length of X_train : ",len(X_train))
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 9,10, 20, 30,40],
        'min_samples_split': [2, 3,5, 8, 10],
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



df = pd.read_csv('metrics/Metric9/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = pd.get_dummies(df, columns=['sector'])
best_model = optimize_random_forest(df)
print(best_model)

"""
{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
"""