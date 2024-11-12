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
from sklearn.dummy import DummyClassifier
import pickle
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)



"""
to run : py -m metrics.Metric4.ml.ml

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

SAVE_MODEL = False
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric4/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(['date', 'symbol'], axis=1)

"""

Feature Importances:
EVEbitdaRatio           0.036778
debtToPrice             0.033291
price                   0.032472
3M_return               0.031130
netDebtToPrice          0.030610
marketCap               0.030488
EV                      0.030458
deriv_2m                0.029231
sma_10d                 0.029021
1Y_6M_growth            0.028328
1y_return               0.026944
sma_50d                 0.026429
sma_10d_1weeks_ago      0.026349
3M_2M_growth            0.026286
6M_return               0.026075
var_sma100D             0.025739
sma_10d_11months_ago    0.025234
sma_100d                0.024947
sma_10d_4months_ago     0.024920
4M_return               0.024683
sma_200d                0.024644
5M_return               0.024337
2M_return               0.023954
sma_10d_1months_ago     0.023479
sma_10d_6months_ago     0.023440
sma_10d_5months_ago     0.023219
sma_10d_3months_ago     0.023199
4M_3M_growth            0.022734
sma_10d_2months_ago     0.022600
2M_1M_growth            0.022085
var_sma50D              0.021968
sma_50w                 0.021548
total_debt              0.020274
1M_return               0.020007
1W_return               0.019508
1M_1W_growth            0.019203
rev_growth              0.019002
deriv_1m                0.018870
var_sma10D              0.017542
netDebt                 0.016816
divpayout_ratio         0.002158
"""


columns_to_use = ['price', 'EVEbitdaRatio', 'peg', 'marketCap', 'sma_10d', 'EV', 
                  'EVGP', 'pe', 'sma_10d_1weeks_ago', 'markRevRAtio', 'sma_50d', 
                  '1Y_6M_growth', 'sma_100d', 'income_lag_days', 'fwdPriceTosale', 
                  'balance_lag_days', '3M_return', 'sma_200d', 'sma_10d_1months_ago', 
                  '1y_return', 'sma_10d_11months_ago', 'sma_10d_3months_ago', 'sma_10d_2months_ago', 
                  'sma_10d_4months_ago', 'min_in_4M', '6M_return', 'sma_10d_7months_ago', 
                  'sma_10d_5months_ago', 'deriv_3m', 'sma_10d_6months_ago', '7M_return', '3M_2M_growth', '5M_return', 'maxPercen_4M', 'max_in_4M', 'sma_50w', 'max_minus_min', '4M_return', 'deriv_2m', 'deriv_1m', 'debtToPrice', 'deriv_1w', 'total_debt', 
                  'var_sma50D_100D', 'var_sma100D', '4M_3M_growth', 'netDebtToPrice', 'var_price_to_minmax', '2M_return', '2M_1M_growth', 'curr_est_eps', 'future_est_rev', 'var_sma10D_100D', 'future_est_eps', 'eps_growth', 'std_10d', '1M_return', 'var_sma50D', 
                  '1W_return', '1M_1W_growth', 'var_sma10D_50D', 'RnD_expenses', 'rev_growth', 'var_sma10D', 'relative_std']

uncorrelated_features= ['divpayout_ratio', 'std_10d', 'RnD_expenses', 'income_lag_days',
       'marketCap', 'total_debt', 'curr_est_eps', 'var_price_to_minmax',
       'maxPercen_4M', 'debtToPrice', 'markRevRAtio', 'eps_growth', 'pe',
       'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'var_sma50D_100D',
       'relative_std', 'deriv_1w', 'deriv_1m', 'deriv_3m', '1W_return',
       '1M_return', '7M_return', '1y_return', '2M_1M_growth', '3M_2M_growth',
       '4M_3M_growth', '1Y_6M_growth']

available_columns = [col for col in columns_to_use if col in df.columns]
available_columns2 = [col for col in uncorrelated_features if col in df.columns]

feature_subsets = [
    available_columns[:16],
    #available_columns[:24],
    #available_columns,
    available_columns2
]



models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Decision Tree md4': DecisionTreeClassifier(random_state=42,max_depth=4),
    'Decision Tree md3': DecisionTreeClassifier(random_state=42,max_depth=3),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Random Forest 200': RandomForestClassifier(n_estimators=200,random_state=42),
    'Random Forest 300': RandomForestClassifier(n_estimators=500,random_state=42),
    #'XGBoost': XGBClassifier(random_state=42),
    #Nonlinear SVM with Gaussian RBF Kernel 
    # Gradient Boosting variants
    #'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    #'Gradient Boosting 200': GradientBoostingClassifier(n_estimators=200, random_state=42),
    #'Gradient Boosting LR01': GradientBoostingClassifier(learning_rate=0.1, random_state=42),
    
     # LightGBM variants
    #'LightGBM': LGBMClassifier(random_state=42),
    #'LightGBM 200': LGBMClassifier(n_estimators=200, random_state=42),
    
    # CatBoost
    #'CatBoost': CatBoostClassifier(random_state=42, verbose=False),
    
    # AdaBoost
    #'AdaBoost': AdaBoostClassifier(random_state=42),
    'XGBoost':XGBClassifier(n_estimators=100),
    'XGBoost 200':XGBClassifier(n_estimators=200),
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
        if precision>0.91 and specificity>0.91:
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