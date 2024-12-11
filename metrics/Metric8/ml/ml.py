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
to run : py -m metrics.Metric8.ml.ml

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
df = pd.read_csv('metrics/Metric8/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(['date', 'symbol'], axis=1)
df = pd.get_dummies(df, columns=['sector'])

favourite_features=  ['est_eps_growth','month','income_lag_days','curr_est_eps','max4M_lag','min4M_lag','max_minus_min','var_price_to_minmax','maxPercen_4M','max1M_lag','min1M_lag','max_minus_min1M','var_price_to_minmax1M','maxPercen_1M','max2W_lag','min2W_lag','max_minus_min2W','var_price_to_minmax2W','maxPercen_2W','max8M_lag','dist_max8M_4M','min8M_lag','dist_min8M_4M','max_minus_min8M','var_price_minmax8M','maxPercen_8M','ebitdaMargin','Spread1Mby8M','Spread4Mby8M','Spread1Mby4M','debtToPrice','markRevRatio','open_price_to_now','eps_growth','pe','peg','PS_to_PEG','netDebtToPrice','dividend_payout_ratio','EVEbitdaRatio','fwdPriceTosale_diff','var_sma50D','var_sma100D','var_sma10D','var_sma10D_50D','var_sma50D_100D','var_sma50D_200D','var_sma10D_100D','var_sma10D_200D','relative_std','deriv_1w','deriv_1m','deriv_2m','deriv_3m','deriv_4m','deriv_5m','deriv_6m','deriv_min4M','deriv_max8M','deriv_min8M','deriv_1w1m_growth','deriv_1m2m_growth','deriv_3m4m_growth','1W_return','price_to_SMA10d_1W','price_to_SMA10d_5M','1M_return','2M_return','3M_return','4M_return','5M_return','6M_return','7M_return','1y_return','1M_1W_growth','2M_1W_growth','1M1W_2M1W_growth','2M_1M_growth','3M_2M_growth','4M_3M_growth','1Y_6M_growth','price_to_sma_10d_ratio','price_to_sma_50d_ratio','price_to_sma_100d_ratio','price_to_sma_200d_ratio','sma_10d_to_sma_50d_ratio','sma_10d_to_sma_100d_ratio','sma_10d_to_sma_200d_ratio','sma_50d_to_sma_100d_ratio','sma_50d_to_sma_200d_ratio','sma_100d_to_sma_200d_ratio','golden_cross','death_cross','smas_above_count','sma10_yoy_growth']
uncorrelated_features1 = ['est_eps_growth', 'month', 'GP', 'netIncome', 'ebitda', 'marketCap',
       'cashcasheq', 'curr_est_eps', 'max_minus_min', 'dist_max8M_4M',
       'dist_min8M_4M', 'ebitdaMargin', 'Spread4Mby8M', 'markRevRatio', 'peg',
       'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale_diff',
       'deriv_min8M', '1y_return', '3M_2M_growth', 'price_to_sma_100d_ratio',
       'sma_10d_to_sma_50d_ratio', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']

uncorrelated_features2 = ['sma_10d_6months_ago', 'marketCap', 'balance_lag_days', 'max_minus_min',
       'min8M_lag', 'Spread4Mby8M', 'markRevRatio', 'peg', 'EVEbitdaRatio',
       'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m',
       'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m', '1y_return',
       'combined_valuation_score']

feature_subsets = [
    
    ['sma_100d', 'sma_200d', 'sma_10d_5months_ago', 'sma_10d_11months_ago',
       'marketCap', 'EV', 'max_minus_min', 'min8M_lag', 'markRevRatio', 'pe',
       'peg', 'EVEbitdaRatio', 'EVGP', 'deriv_1m', 'deriv_2m', 'deriv_3m',
       'deriv_4m', 'deriv_5m', 'combined_valuation_score', 'sma10_yoy_growth'],

       ['price', 'sma_10d_11months_ago', 'month', 'netIncome', 'total_debt',
       'cashcasheq', 'netDebt', 'curr_est_eps', 'EV', 'max_minus_min',
       'dist_max8M_4M', 'dist_min8M_4M', 'ebitdaMargin', 'Spread4Mby8M',
       'markRevRatio', 'pe', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP',
       'fwdPriceTosale', 'fwdPriceTosale_diff', 'var_sma50D', 'deriv_min8M',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sector_energy'],

    ['sma_200d', 'EV', 'pe', 'peg', 'EVGP', 'deriv_2m', 'deriv_4m',
       'combined_valuation_score'],
        uncorrelated_features1,
        uncorrelated_features2,
['sma_200d', 'sma_10d_5months_ago', 'marketCap', 'pe', 'EVGP',
       'deriv_2m', 'combined_valuation_score', 'sma10_yoy_growth'],

       ['sma_200d', 'sma_50d', 'sma_10d_5months_ago', 'marketCap',
       'max_minus_min', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'deriv_1m',
       'deriv_2m', 'deriv_4m', 'combined_valuation_score', 'sma10_yoy_growth'],

['sma_200d', 'sma_50d', 'sma_10d_5months_ago', 'sma_10d_11months_ago',
'income_lag_days', 'marketCap', 'EV', 'max_minus_min', 'markRevRatio',
'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'deriv_1m',
'deriv_2m', 'deriv_4m', 'deriv_5m', 'combined_valuation_score',
'sma10_yoy_growth'],

['sma_50w', 'sma_10d_1months_ago', 'sma_10d_5months_ago', 'month',
       'revenues', 'GP', 'netIncome', 'cashcasheq', 'netDebt', 'curr_est_eps',
       'max_minus_min', 'dist_min8M_4M', 'Spread4Mby8M', 'markRevRatio', 'peg',
       'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff',
       'var_sma50D_100D', 'var_sma50D_200D', 'deriv_min4M', '2M_1W_growth'],

['month', 'ebitda', 'cashcasheq', 'netDebt', 'curr_est_eps',
       'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M', 'Spread4Mby8M',
       'debtToPrice', 'PS_to_PEG', 'dividend_payout_ratio', 'EVEbitdaRatio',
       'fwdPriceTosale_diff', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score'],

['sma_200d', 'sma_10d_5months_ago', 'sma_10d_11months_ago',
       'income_lag_days', 'marketCap', 'EV', 'max_minus_min', 'min8M_lag',
       'Spread4Mby8M', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues',
       'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m',
       'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m',
       'combined_valuation_score', 'sma10_yoy_growth'],

['sma_10d_5months_ago', 'month', 'ebitda', 'cashcasheq', 'netDebt',
       'curr_est_eps', 'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M',
       'Spread4Mby8M', 'debtToPrice', 'peg', 'PS_to_PEG',
       'dividend_payout_ratio', 'EVEbitdaRatio', 'fwdPriceTosale_diff',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score'],
['price', 'sma_100d', 'sma_200d', 'sma_50d', 'sma_10d_4months_ago', 'sma_10d_6months_ago', 'marketCap', 'EV', 'max_minus_min8M', 'debtToPrice', 'pe', 'peg', 'netDebtToPrice', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'deriv_4m', 'deriv_5m', 'deriv_6m', 'sma_50d_to_sma_200d_ratio', 'combined_valuation_score', 'sma10_yoy_growth'],
['price', 'est_eps_growth', 'sma_100d', 'sma_200d', 'month', 'GP', 'netIncome', 'marketCap', 'cashcasheq', 'EV', 'dist_max8M_4M', 'markRevRatio', 'netDebtToPrice', 'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_6m', 'deriv_min8M'],
['price', 'sma_100d', 'RnD_expenses', 'EV', 'maxPercen_4M', 'max_in_2W', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', '3M_2M_growth', '1Y_6M_growth', 'combined_valuation_score', 'sma10_yoy_growth'],
['EVEbitdaRatio', 'sma_200d', 'sma_50d', 'marketCap', 'markRevRatio', 'EVRevenues', 'fwdPriceTosale', 'deriv_2m', '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score', 'sma10_yoy_growth'],
['price', 'month', 'RnD_expenses', 'revenues', 'GP', 'dividendsPaid', 'curr_est_eps', 'EV', 'min_in_4M', 'max_minus_min', 'max_in_8M', 'dist_min8M_4M', 'ebitdaMargin', 'eps_growth', 'peg', 'PS_to_PEG', 'fwdPriceTosale_diff', 'var_sma50D_100D', '1Y_6M_growth', 'combined_valuation_score'],
]



models = {
    'LR': LogisticRegression(),
    'DT1': DecisionTreeClassifier(),
    'DT2': DecisionTreeClassifier(random_state=42,max_depth=4),
    'DT3': DecisionTreeClassifier(random_state=42,max_depth=3),
    'SVC': SVC(kernel='linear', C=1.0),
    'SVC2': SVC(kernel='rbf', C=1.0,class_weight = None, degree = 2,gamma = 'scale'),
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
    'RF13': RandomForestClassifier(n_estimators= 300, max_features=None, min_samples_leaf= 1, min_samples_split= 2,max_depth= None),

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
     'XGB12':XGBClassifier(n_estimators=200,colsample_bytree= 0.9,gamma=0,learning_rate=0.3,max_depth=7,min_child_weight=1,subsample=0.8),
       'XGB13':XGBClassifier(n_estimators=200,colsample_bytree= 0.7,gamma=0.2,learning_rate=0.3,max_depth=8,min_child_weight=1,subsample=0.8),

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
        if (precision>0.9 and specificity>0.9) or (precision>0.85 and specificity>=0.98):
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