import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
import pickle
from sklearn.model_selection import GridSearchCV

"""
to run : py -m metrics.Metric3.ml.grid_search_xgboost
"""

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(['date', 'symbol'], axis=1)

columns_to_use2 = [
         'peRatio',
'ratio' ,             
'MarkRevRatio'  ,     
'price'     ,         
'sma_200d'   ,        
'sma_50w'   ,         
'fcf'      ,          
'sma_100d'  ,         
'ocfToNetinc' ,       
'price_to_50w' ,      
'sma_10d'    ,        
'cashAtEndOfPeriod'  ,
'other_expenses'  ,   
'sma50_to_sma200' ,   
'price_to_sma200' ,   
'revenues'        ,   
'eps'             ,   
'sma50_to_50w'    ,   
'ebitdaMargin'    ,   
'NetProfitMargin' ,   
'price_to_sma100' ,   
'otherExpRevenue' ,   
'momentum_score'  ,   
'sma10_to_sma100' ,   
'costOfRevenue'   ,   
'revenues_lag_days' , 
'price_to_sma50'   ,  
'cost_revenue_ratio' ,
'sma10_to_sma50'  ,   
'price_to_sma10'  ,   
]

available_columns2 = [col for col in columns_to_use2 if col in df.columns]
X = df[available_columns2]
y = df['to_buy']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create the XGBoost classifier
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Define the parameter grid
param_grid = {
    'max_depth': [ 9,10],
    'learning_rate': [0.08,0.09, 0.1],
    'n_estimators': [200],
    'min_child_weight': [0.7,0.8,1],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0]
}

# Create scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring=scoring,
    refit='accuracy',  # You can change this to optimize for a different metric
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))

# Get feature importance of the best model
feature_importance = pd.DataFrame({
    'feature': available_columns2,
    'importance': grid_search.best_estimator_.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)


"""
Best parameters: {'gamma': 0, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 0.1, 'subsample': 0.8}
Best cross-validation accuracy: 0.9712400667033101

Test Set Performance:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1490
           1       0.98      0.98      0.98      1501

    accuracy                           0.98      2991
   macro avg       0.98      0.98      0.98      2991
weighted avg       0.98      0.98      0.98      2991


Feature Importance:
            feature  importance
12              eps    0.085491
13  NetProfitMargin    0.079058
8          revenues    0.078252
4           sma_50w    0.073697
7    other_expenses    0.073363
14     ebitdaMargin    0.072077
10     price_to_50w    0.057477
17  otherExpRevenue    0.054646
6             price    0.053368
0      MarkRevRatio    0.051453
5          sma_100d    0.045097
2          sma_200d    0.044561
11  sma50_to_sma200    0.041679
1             ratio    0.041477
3           peRatio    0.040166
15     sma50_to_50w    0.040047
16  price_to_sma200    0.038042
9           sma_10d    0.030050



Best parameters: {'gamma': 0.1, 'learning_rate': 0.09, 'max_depth': 8, 'min_child_weight': 0.8, 'n_estimators': 200, 'reg_lambda': 0.1, 'subsample': 0.8}
Best cross-validation accuracy: 0.9699027088127246

Test Set Performance:
              precision    recall  f1-score   support

           0       0.97      0.98      0.97      1473
           1       0.98      0.97      0.97      1518

    accuracy                           0.97      2991
   macro avg       0.97      0.97      0.97      2991
weighted avg       0.97      0.97      0.97      2991


Feature Importance:
            feature  importance
12              eps    0.084023
17  otherExpRevenue    0.076675
4           sma_50w    0.073243
8          revenues    0.072738
13  NetProfitMargin    0.071492
14     ebitdaMargin    0.068374
7    other_expenses    0.068310
2          sma_200d    0.054226
10     price_to_50w    0.053389
11  sma50_to_sma200    0.048608
16  price_to_sma200    0.044318
5          sma_100d    0.044271
0      MarkRevRatio    0.043630
3           peRatio    0.043080
15     sma50_to_50w    0.042347
1             ratio    0.042118
6             price    0.041044
9           sma_10d    0.028114
"""