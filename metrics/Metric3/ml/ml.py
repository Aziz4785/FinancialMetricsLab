import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
import pickle
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


"""
to run : py -m metrics.Metric3.ml.ml

YOU MUST SELECT MANY MODELS (with different architecture and different input features)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WE WANT TO AVOID : 0s predicted as 1s
IT DOESN'T MATTER IF A 1 IS PREDICTED AS 0
BUT A 0 SHOULD NEVER NEVER BE PREDICTED AS 1

=> we should maximize the PRECISION and SPECIFICITY

1 predicted as 1 -> true positive
0 predicted as 0 -> true negative
0 pedicted as 1 -> false positive
1 predicted as 0 -> false negative

we want to minimize false positive

specificity: tn/(tn+fp)  (True Negative Rate (TNR))
precision = tp/(tp+fp)
recall = tp/(tp+fn)
!!!!!!!!!!!!!!!!!s!!!!!!!!!!!!!!!!!!!!!
"""

SAVE_MODEL = True
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(['date', 'symbol'], axis=1)


"""
Feature Importances:
peRatio               0.052717
ratio                 0.051066
MarkRevRatio          0.048813
price                 0.047952
sma_200d              0.047670
sma_50w               0.047183
fcf                   0.045524
sma_100d              0.044764
ocfToNetinc           0.039141
price_to_50w          0.038214
sma_10d               0.037762
cashAtEndOfPeriod     0.037601
other_expenses        0.035626
sma50_to_sma200       0.035100
price_to_sma200       0.034634
revenues              0.032978
eps                   0.032386
sma50_to_50w          0.032052
ebitdaMargin          0.030785
NetProfitMargin       0.030220
price_to_sma100       0.025756
otherExpRevenue       0.025629
momentum_score        0.023170
sma10_to_sma100       0.023049
costOfRevenue         0.022051
revenues_lag_days     0.019236
price_to_sma50        0.017823
cost_revenue_ratio    0.017104
sma10_to_sma50        0.015148
price_to_sma10        0.008847
"""

columns_to_use = [
 'price',
'MarkRevRatio' ,
'sma_200d' ,
'ratio' ,
'fcf',
'sma_50w' ,
'sma_100d' ,
'sma_10d' ,
'other_expenses'   ,
'eps',
'revenues'   ,
'ebitdaMargin'  ,
'NetProfitMargin'   ,
'otherExpRevenue'    ,
'costOfRevenue'  ,
'revenues_lag_days' ,
'cost_revenue_ratio'
]
columns_to_use2 = [
'MarkRevRatio' ,  
'ratio'  ,        
'sma_200d' ,      
'peRatio'  ,      
'sma_50w'  ,      
'sma_100d' ,      
'price'   ,       
'other_expenses' ,
'revenues' ,      
'sma_10d' ,       
'price_to_50w' ,  
'sma50_to_sma200',
'eps'  ,           
'NetProfitMargin' ,
'ebitdaMargin',    
'sma50_to_50w' ,   
'price_to_sma200' ,
'otherExpRevenue' ,
]
columns_to_use3 = [
         'peRatio',
'ratio',              
'MarkRevRatio'  ,     
'price' ,             
'sma_200d',           
'sma_50w' ,           
'fcf'   ,             
'sma_100d',           
'ocfToNetinc' ,       
'price_to_50w' ,      
'sma_10d'   ,         
'cashAtEndOfPeriod' , 
'other_expenses' ,    
'sma50_to_sma200' ,   
'price_to_sma200' ,   
'revenues'   ,        
'eps'  ,              
'sma50_to_50w' ,      
'ebitdaMargin',       
'NetProfitMargin' ,   
'price_to_sma100' ,   
'otherExpRevenue' ,   
'momentum_score' ,    
'sma10_to_sma100' ,   
'costOfRevenue' ,     
'revenues_lag_days' , 
'price_to_sma50' ,    
'cost_revenue_ratio' ,
'sma10_to_sma50' ,    
'price_to_sma10' ,    
]

columns_to_use5 = [
                 'fcf', 
        'price_to_50w' ,
             'sma_50w' ,
            'revenues' ,
      'other_expenses' ,
            'sma_200d' ,
   'cashAtEndOfPeriod' ,
     'sma50_to_sma200', 
  'cost_revenue_ratio', 
                 'eps' ,
         'ocfToNetinc' ,
     'otherExpRevenue' ,
     'NetProfitMargin' ,
            'sma_100d' ,
       'costOfRevenue' ,
             'peRatio' ,
        'ebitdaMargin' ,
     'price_to_sma200' ,
        'MarkRevRatio' ,
        'sma50_to_50w' ,
               'ratio' ,
     'price_to_sma100' ,
               'price' ,
     'sma10_to_sma100' ,
             'sma_10d' ,
      'sma10_to_sma50' ,
      'price_to_sma50' ,
      'momentum_score' ,
   'revenues_lag_days' ,
      'price_to_sma10' ,
]




available_columns = [col for col in columns_to_use if col in df.columns]
available_columns2 = [col for col in columns_to_use2 if col in df.columns]
available_columns3 = [col for col in columns_to_use3 if col in df.columns]
available_columns5 = [col for col in columns_to_use5 if col in df.columns]

feature_subsets = [
    available_columns[:2],
    available_columns[:3],
    available_columns[:4],
    available_columns[:5],
    available_columns[:6],
    available_columns[:7],
    available_columns[:9],
    available_columns[:10],
    available_columns[:11],
    available_columns2[:3],
    available_columns2[:4],
    available_columns2[:5],
    available_columns2[:6],
    available_columns2[:7],
    available_columns2[:8],
    available_columns2[:9],
    available_columns2[:10],
    available_columns2[:11],
    available_columns3[:4],
    available_columns3[:6],
    available_columns3[:8],
    available_columns3[:10],
    available_columns3[:11],
    available_columns3[:13],
    available_columns5[:4],
    available_columns5[:6],
    available_columns5[:8],
    available_columns5[:10],
    available_columns5[:11],
    available_columns5[:13],
]



models = {
    #'Logistic Regression': LogisticRegression(),
    #'Decision Tree': DecisionTreeClassifier(random_state=42),
    #'Decision Tree md4': DecisionTreeClassifier(random_state=42,max_depth=4),
    #'Decision Tree md3': DecisionTreeClassifier(random_state=42,max_depth=3),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Random Forest 200': RandomForestClassifier(n_estimators=200,random_state=42),
    'Random Forest 300': RandomForestClassifier(n_estimators=500,random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    Nonlinear SVM with Gaussian RBF Kernel 
    # Gradient Boosting variants
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    #'Gradient Boosting 200': GradientBoostingClassifier(n_estimators=200, random_state=42),
    #'Gradient Boosting LR01': GradientBoostingClassifier(learning_rate=0.1, random_state=42),
    
     # LightGBM variants
    #'LightGBM': LGBMClassifier(random_state=42),
    #'LightGBM 200': LGBMClassifier(n_estimators=200, random_state=42),
    
    # CatBoost
    #'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
    
    # AdaBoost
    #'AdaBoost': AdaBoostClassifier(random_state=42),

    'XGBoost 200':XGBClassifier(n_estimators=200),
    'XGBoost c1':XGBClassifier(colsample_bytree= 1.0, gamma= 0.1, learning_rate= 0.1, max_depth= 6, min_child_weight= 1, n_estimators= 200, subsample= 0.9),
    'XGBoost c2':XGBClassifier(gamma=0, learning_rate= 0.1, max_depth= 7, min_child_weight= 0.8, n_estimators= 200, reg_lambda= 0.5, subsample= 0.8),
    'XGBoost c3':XGBClassifier(gamma= 0.1, learning_rate= 0.09, max_depth= 8, min_child_weight= 0.8, n_estimators= 200, reg_lambda= 0.1, subsample= 0.8),
    'XGBoost c4':XGBClassifier(gamma= 0, learning_rate= 0.1, max_depth= 9, min_child_weight= 1, n_estimators= 200, reg_lambda= 0.1, subsample= 0.8),

    #'Dummy (Always 0)': DummyClassifier(strategy='constant', constant=0)
}


for i, subset in enumerate(feature_subsets):
    print(f"\n========== Feature Subset {i}: {subset} ==========")
    X = df[subset]
    y = df['to_buy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    good_model_found=False

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        precision = precision_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        if precision>0.97 and specificity>0.98:
            good_model_found=True
            print("----------------------")
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{name} Results:")
            
            print(f"specificity = {specificity}")
            #print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision Score: {precision:.4f}")
            print("Classification Report:")
            classification_rep = classification_report(y_test, y_pred)
            print(classification_rep)
            #print(classification_rep[0])
            if SAVE_MODEL:
                if name == 'XGBoost':
                    with open(f'xgboost_model_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'XGBoost 200':
                    with open(f'xgboost200_model_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'Random Forest':
                    with open(f'rf_100_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'Random Forest 200':
                    with open(f'rf_200_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'Random Forest 300':
                    with open(f'rf_300_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'XGBoost c1':
                    with open(f'xg_c1_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'XGBoost c2':
                    with open(f'xg_c2_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)
                
                elif name == 'XGBoost c3':
                    with open(f'xg_c3_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)

                elif name == 'XGBoost c4':
                    with open(f'xg_c4_{i}_{len(subset)}.pkl', 'wb') as model_file:
                        pickle.dump(model, model_file)
    if SAVE_MODEL and good_model_found:
        with open(f'scaler_{i}_{len(subset)}.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)