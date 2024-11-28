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
import os
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)



"""
to run : py -m metrics.Metric6.ml.ml

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
folder = 'allmodels'
os.makedirs(folder, exist_ok=True)
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric6/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(['date', 'symbol'], axis=1)


feature_subsets = [
    ['price', 'sma_100d', 'pe', 'peg', 'EVGP', '1Y_6M_growth']

    ,['price', 'sma_100d', 'RnD_expenses', 'EV', 'maxPercen_4M', 'max_in_2W',
        'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale',
        'fwdPriceTosale_diff', '3M_2M_growth', '1Y_6M_growth',
        'combined_valuation_score', 'sma10_yoy_growth']

    ,['price', 'pe_ratio', 'sma_100d', 'RnD_expenses', 'marketCap', 'EV',
        'maxPercen_4M', 'max_in_2W', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP',
        'fwdPriceTosale', 'fwdPriceTosale_diff', '3M_2M_growth', '1Y_6M_growth',
        'combined_valuation_score', 'sma10_yoy_growth']

    ,['RnD_expenses', 'revenues', 'GP', 'max_in_8M', 'ebitdaMargin',
       'eps_growth']

    ,['price', 'month', 'RnD_expenses', 'revenues', 'GP', 'curr_est_eps',
        'min_in_4M', 'max_minus_min', 'max_in_8M', 'dist_min8M_4M',
        'ebitdaMargin', 'eps_growth', 'peg', 'PS_to_PEG', '1Y_6M_growth',
        'combined_valuation_score']

    ,['price', 'month', 'RnD_expenses', 'revenues', 'GP', 'dividendsPaid',
        'curr_est_eps', 'EV', 'min_in_4M', 'max_minus_min', 'max_in_8M',
        'dist_min8M_4M', 'ebitdaMargin', 'eps_growth', 'peg', 'PS_to_PEG',
        'fwdPriceTosale_diff', 'var_sma50D_100D', '1Y_6M_growth',
        'combined_valuation_score'],

]



models = {
    'LR': LogisticRegression(),
    'DT1': DecisionTreeClassifier(),
    'DT2': DecisionTreeClassifier(random_state=42,max_depth=4),
    'DT3': DecisionTreeClassifier(random_state=42,max_depth=3),
    'RF': RandomForestClassifier(),
    'RF1': RandomForestClassifier(n_estimators=200,),
    'RF2': RandomForestClassifier(n_estimators=300,),
    'RF3': RandomForestClassifier(n_estimators=600,),
   'RF4a': RandomForestClassifier(n_estimators=100,max_features=None,min_samples_leaf=1,min_samples_split=2),
    'RF4b': RandomForestClassifier(n_estimators=100,max_features=None,min_samples_leaf=1,min_samples_split=2,max_depth= 20),
    'RF4c': RandomForestClassifier(n_estimators=300,max_features='sqrt',min_samples_leaf=1,min_samples_split=2),
    'RF4': RandomForestClassifier(n_estimators=100,max_features='sqrt',min_samples_leaf=1,min_samples_split=2),
    'RF5': RandomForestClassifier(n_estimators=200,max_features='sqrt',min_samples_leaf=1,min_samples_split=2),
    'RF6': RandomForestClassifier(n_estimators=300,max_features=None,min_samples_leaf=1,min_samples_split=2),
    'RF7': RandomForestClassifier(n_estimators=200,max_features=None,min_samples_leaf=1,min_samples_split=2),
    'RF8': RandomForestClassifier(n_estimators=100,max_features= None, min_samples_leaf= 1, min_samples_split= 5,max_depth=10),
    'RF9': RandomForestClassifier(n_estimators= 100, max_features='log2', min_samples_leaf= 1, min_samples_split= 2,max_depth= None,),
    'RF10': RandomForestClassifier(n_estimators= 100, max_features='sqrt', min_samples_leaf= 1, min_samples_split= 2,max_depth= 10),
    'RF11': RandomForestClassifier(n_estimators= 300, max_features='sqrt', min_samples_leaf= 1, min_samples_split= 2,max_depth= 10),
    'RF12': RandomForestClassifier(n_estimators= 200, max_features=None, min_samples_leaf= 1, min_samples_split= 2,max_depth= 10),


    'XGB1':XGBClassifier(n_estimators=100),
    'XGB2':XGBClassifier(n_estimators=200),
    'XGB3':XGBClassifier(n_estimators=200,eval_metric='logloss'),
    'XGB4':XGBClassifier(n_estimators=200,colsample_bytree= 0.7,gamma=0,learning_rate=0.3,max_depth=5,min_child_weight=1,subsample=0.8),
    'XGB5':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0.1,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB6':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0.2,learning_rate=0.3,max_depth=5,min_child_weight=1,subsample=0.8),
    'XGB7':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.1,max_depth=5,min_child_weight=1,subsample=0.9),
    'XGB8':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0,learning_rate=0.3,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB9':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB10':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.3,max_depth=7,min_child_weight=3,subsample=0.9),
    'XGB11':XGBClassifier(n_estimators=200,colsample_bytree= 0.9,gamma=0.1,learning_rate=0.3,max_depth=7,min_child_weight=1,subsample=0.9),


    #'Dummy (Always 0)': DummyClassifier(strategy='constant', constant=0)
}


for i, subset in enumerate(feature_subsets):
    print(f"\n========== Feature Subset {i}: {subset} ==========")
    X = df[subset]
    y = df['to_buy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
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
        if precision>0.98 and specificity>0.98:
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
                with open(f'{folder}/{name}_{i}_{len(subset)}.pkl', 'wb') as model_file:
                    pickle.dump(model, model_file)

    if SAVE_MODEL and good_model_found:
        with open(f'{folder}/scaler_{i}_{len(subset)}.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)