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
to run : py -m metrics.Metric5.ml.ml

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
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(['date', 'symbol'], axis=1)


feature_subsets = [
    ['price', 'evebitda', 'sma_50d', 'marketCap', 'EV', 'max_in_2W',
       'markRevRatio', 'peg', 'EVRevenues', 'fwdPriceTosale',
       'fwdPriceTosale_diff', 'var_sma100D', 'var_sma50D_100D',
       'var_sma10D_100D', '1y_return', '1Y_6M_growth']
    ,
    ['evebitda', 'sma_10d', 'marketCap', 'peg', 'EVRevenues', '1y_return']
    ,
    ['evebitda', 'sma_10d', 'sma_50d', 'marketCap', 'EV', 'max_minus_min8M',
       'markRevRatio', 'peg', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'var_sma100D', 'var_sma50D_100D', 'var_sma10D_100D', '1y_return',
       '1Y_6M_growth']
    ,
    ['evebitda', 'price', 'sma_50d', 'marketCap', 'EV', 'max_minus_min8M',
       'markRevRatio', 'peg', 'EVGP', 'fwdPriceTosale',
       'var_sma100D', 'var_sma50D_100D', 'var_sma10D_100D', '1y_return',
       '1Y_6M_growth']
    ,
    ['price', 'evebitda', 'marketCap', 'markRevRatio', 'peg', 'EVRevenues',
       'fwdPriceTosale', 'var_sma50D_100D', 'var_sma10D_100D', '1y_return']
    ,

    ['RnD_expenses', 'eps_growth', 'deriv_min8M',
       'sma_100d_to_sma_200d_ratio'],

    ['evebitda', 'RnD_expenses', 'marketCap', 'eps_growth', 'peg',
       'EVRevenues', 'fwdPriceTosale', 'var_sma10D_100D', 'deriv_2m',
       'deriv_max4M', 'deriv_min8M', '1Y_6M_growth',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score'],
    
    ['evebitda', 'RnD_expenses', 'marketCap', 'eps_growth', 'peg',
       'fwdPriceTosale', 'var_sma10D_100D', 'deriv_2m', 'deriv_min8M',
       '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']   ,
    
    ['RnD_expenses', 'eps_growth', 'peg', 'var_sma10D_100D', 'deriv_min8M',
       '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']  ,

    ['marketCap', 'fwdPriceTosale', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']
    
    ,['evebitda', 'sma_50d', 'marketCap', 'markRevRatio', 'EVRevenues',
       'fwdPriceTosale', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']
    
    ,['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'markRevRatio',
       'EVRevenues', 'fwdPriceTosale', 'deriv_2m', '1Y_6M_growth',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth']
    
    ,['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'EV', 'markRevRatio',
       'peg', 'netDebtToPrice', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'deriv_2m', '4M_return', '1Y_6M_growth', 'sma_50d_to_sma_100d_ratio',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth']
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
    'RF4': RandomForestClassifier(n_estimators=100,max_features='sqrt',min_samples_leaf=1,min_samples_split=2),
    'RF5': RandomForestClassifier(n_estimators=200,max_features='sqrt',min_samples_leaf=1,min_samples_split=2),
    'RF6': RandomForestClassifier(n_estimators=300,max_features=None,min_samples_leaf=1,min_samples_split=2),
    'RF7': RandomForestClassifier(n_estimators=200,max_features=None,min_samples_leaf=1,min_samples_split=2),
    'RF8': RandomForestClassifier(n_estimators=100,max_features= None, min_samples_leaf= 1, min_samples_split= 5,max_depth=10),
    'RF9': RandomForestClassifier(n_estimators= 100, max_features='log2', min_samples_leaf= 1, min_samples_split= 2,max_depth= None,),
    'RF10': RandomForestClassifier(n_estimators= 100, max_features='sqrt', min_samples_leaf= 1, min_samples_split= 2,max_depth= 10),
    'RF11': RandomForestClassifier(n_estimators= 300, max_features='sqrt', min_samples_leaf= 1, min_samples_split= 2,max_depth= 10),

    'XGB1':XGBClassifier(n_estimators=100),
    'XGB2':XGBClassifier(n_estimators=200),
    'XGB3':XGBClassifier(n_estimators=200,eval_metric='logloss'),
    'XGB4':XGBClassifier(n_estimators=200,colsample_bytree= 0.7,gamma=0,learning_rate=0.3,max_depth=5,min_child_weight=1,subsample=0.8),
    'XGB5':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0.1,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB6':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0.2,learning_rate=0.3,max_depth=5,min_child_weight=1,subsample=0.8),
    'XGB7':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.1,max_depth=5,min_child_weight=1,subsample=0.9),
    'XGB8':XGBClassifier(n_estimators=300,colsample_bytree= 0.9,gamma=0,learning_rate=0.3,max_depth=7,min_child_weight=1,subsample=0.7),
    'XGB9':XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0,learning_rate=0.1,max_depth=7,min_child_weight=1,subsample=0.7),
    

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
        if precision>0.989 and specificity>0.989:
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
                with open(f'{name}_{i}_{len(subset)}.pkl', 'wb') as model_file:
                    pickle.dump(model, model_file)

    if SAVE_MODEL and good_model_found:
        with open(f'scaler_{i}_{len(subset)}.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)