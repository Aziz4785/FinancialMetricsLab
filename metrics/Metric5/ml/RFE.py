import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
"""
to run : py -m metrics.Metric5.ml.RFE
"""
"""
best subset of features so far (for random forest : ) :
Index(['marketCap', 'fwdPriceTosale', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score'],
      dtype='object')
Accuracy: 0.967741935483871

Selected Features:
Index(['evebitda', 'sma_50d', 'marketCap', 'markRevRatio', 'EVRevenues',
       'fwdPriceTosale', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score'],
      dtype='object')
Accuracy: 0.9741935483870968

Index(['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'markRevRatio',
       'EVRevenues', 'fwdPriceTosale', 'deriv_2m', '1Y_6M_growth',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth'],
      dtype='object')
Accuracy: 0.9870967741935484


Selected Features:
Index(['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'EV', 'markRevRatio',
       'peg', 'netDebtToPrice', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'deriv_2m', '4M_return', '1Y_6M_growth', 'sma_50d_to_sma_100d_ratio',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth'],
      dtype='object')
Accuracy: 0.9870967741935484


best subset of features so far (for xgboost : ) :

['RnD_expenses', 'eps_growth', 'deriv_min8M',
       'sma_100d_to_sma_200d_ratio']

['RnD_expenses', 'eps_growth', 'peg', 'var_sma10D_100D', 'deriv_min8M',
       '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']   
['evebitda', 'RnD_expenses', 'marketCap', 'eps_growth', 'peg',
       'fwdPriceTosale', 'var_sma10D_100D', 'deriv_2m', 'deriv_min8M',
       '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']   
['evebitda', 'RnD_expenses', 'marketCap', 'eps_growth', 'peg',
       'EVRevenues', 'fwdPriceTosale', 'var_sma10D_100D', 'deriv_2m',
       'deriv_max4M', 'deriv_min8M', '1Y_6M_growth',
       'sma_100d_to_sma_200d_ratio', 'combined_valuation_score']
"""
"""
these variables are almost the same (correlation>=0.99): 

"""
MODEL_SELECTED = 'XGBOOST' #'RF' or 'XGBOOST'


df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

df = pd.get_dummies(df, columns=['sector'])

X = df.drop(columns=['to_buy', 'date', 'symbol'], errors='ignore')
y = df['to_buy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1)
xgb_model = XGBClassifier(n_estimators=400, random_state=42, eval_metric='logloss', n_jobs=-1)

for k in range(4,20,2):
    print(f"-----number of features : {k}")
    if MODEL_SELECTED == 'RF':
        RFE_selector = RFE(estimator=rf_model, n_features_to_select=k, step=1)
    else : 
        RFE_selector = RFE(estimator=xgb_model, n_features_to_select=k, step=1)
    RFE_selector.fit(X_train_scaled, y_train)

    # Get the boolean mask of selected features
    selected_features_mask = RFE_selector.support_

    # Get the ranking of features
    feature_ranking = RFE_selector.ranking_

    # Print the selected features
    selected_features = X.columns[selected_features_mask]
    print("Selected Features:")
    print(selected_features)

    # Optionally, you can also see feature rankings
    # feature_ranking_df = pd.DataFrame({
    #     'Feature': X.columns,
    #     'Ranking': feature_ranking
    # }).sort_values(by='Ranking')
    # print("\nFeature Rankings:")
    # print(feature_ranking_df)

    # Transform the training and testing data to contain only the selected features
    X_train_selected = RFE_selector.transform(X_train_scaled)
    X_test_selected = RFE_selector.transform(X_test_scaled)

    if MODEL_SELECTED == 'RF':
        rf_model.fit(X_train_selected, y_train)
        y_pred = rf_model.predict(X_test_selected)
    else:
        xgb_model.fit(X_train_selected, y_train)
        y_pred = xgb_model.predict(X_test_selected)

    # Evaluate the model
    #print("Model Performance with RFE-selected Features:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))