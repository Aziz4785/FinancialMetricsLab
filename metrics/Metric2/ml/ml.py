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


"""
to run : py -m metrics.Metric2.ml.ml
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WE WANT TO AVOID : 0s predicted as 1s
IT DOESN'T MATTER IF A 1 IS PREDICTED AS 0
BUT A 0 SHOULD NEVER NEVER BE PREDICTED AS 1

=> we should maximize the PRECISION

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
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric2/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
# Remove 'date' and 'symbol' columns


df = df.drop(['date', 'symbol'], axis=1)

# Separate features (X) and target variable (y)

"""
Feature Importances:
revenues              0.087968
pe_ratio              0.076174
ebitdaMargin          0.071780
price                 0.070905
sma_100d              0.067711
ratio                 0.065914
sma_200d              0.064920
costOfRevenue         0.063804
sma_50w               0.058572
sma_10d               0.056494
sma_50d               0.055273
cost_revenue_ratio    0.054615
other_expenses        0.053977
NetProfitMargin       0.042326
netIncome             0.041542
eps                   0.040213
revenues_lag_days     0.027815
"""
columns_to_use =[          
'price',             
'pe_ratio',          
'ratio',            
'sma_100d', 
'revenues' ,         
'sma_10d',           
'sma_200d' ,         
'sma_50w',           
'sma_50d',           
'costOfRevenue',      
'other_expenses' ,   
'eps' ,              
'netIncome' ,        
'ebitdaMargin' ,'cost_revenue_ratio','NetProfitMargin' ,'revenues_lag_days']             

columns_to_use2 =[   
'revenues' ,        
'price',             
'pe_ratio', 
'ebitdaMargin',         
'ratio',            
'sma_100d',  
'sma_200d' ,    
'sma_10d',           
'sma_50w',           
'sma_50d',           
'costOfRevenue',    
'other_expenses' ,   
'eps' , 'netIncome' , 'cost_revenue_ratio','NetProfitMargin','revenues_lag_days' ]     


available_columns = [col for col in columns_to_use if col in df.columns]
available_columns2 = [col for col in columns_to_use2 if col in df.columns]

feature_subsets = [
    available_columns[:2],
    available_columns[:3],
    available_columns[:4],
    available_columns[:5],
    available_columns[:6],
    available_columns[:7],
    available_columns[:9],
    available_columns2[:4],
    available_columns2[:5],
    available_columns2[:6],
    available_columns2[:7]
]


X = df[available_columns]
y = df['to_buy']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    #'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Decision Tree md4': DecisionTreeClassifier(random_state=42,max_depth=4),
    'Decision Tree md3': DecisionTreeClassifier(random_state=42,max_depth=3),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Random Forest 200': RandomForestClassifier(n_estimators=200,random_state=42),
    'Random Forest 300': RandomForestClassifier(n_estimators=300,random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    #'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
    #'Neural Network 200': MLPClassifier(hidden_layer_sizes=(200,), random_state=42, max_iter=1000),
    'XGBoost 200':XGBClassifier(n_estimators=200),
    'Dummy (Always 0)': DummyClassifier(strategy='constant', constant=0)
}

for i, subset in enumerate(feature_subsets):
    print(f"\n========== Feature Subset {i}: {subset} ==========")
    X = df[subset]
    y = df['to_buy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        precision = precision_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        if precision>0.979 and specificity>0.979:
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

    if SAVE_MODEL:
        with open(f'scaler_{i}_{len(subset)}.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)


# Function to predict 'to_buy' for new data
def predict_to_buy(price, sma_50, gross_profit):
    return model.predict([[price, sma_50, gross_profit]])[0]

