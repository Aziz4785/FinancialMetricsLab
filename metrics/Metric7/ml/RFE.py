import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
"""
to run : py -m metrics.Metric7.ml.RFE
"""
"""
best subset of features so far (for random forest : ) :

['sma_50w', 'sma_200d', 'marketCap', 'netDebtToPrice', 'EVEbitdaRatio',
       'deriv_6m']

['sma_200d', 'marketCap', 'peg', 'netDebtToPrice', 'EVEbitdaRatio',
       'deriv_6m']
['sma_200d', 'sma_10d_6months_ago', 'marketCap', 'peg', 'netDebtToPrice',
       'EVEbitdaRatio', 'EVGP', 'deriv_6m']

['sma_200d', 'sma_50d', 'sma_10d_4months_ago', 'sma_10d_6months_ago',
       'marketCap', 'EV', 'max_minus_min8M', 'debtToPrice', 'peg',
       'netDebtToPrice', 'EVEbitdaRatio', 'EVGP', 'deriv_4m', 'deriv_6m',
       'sma_50d_to_sma_200d_ratio', 'combined_valuation_score']

['price', 'sma_100d', 'sma_200d', 'sma_50d', 'sma_10d_4months_ago',
       'sma_10d_6months_ago', 'marketCap', 'EV', 'max_minus_min8M',
       'debtToPrice', 'pe', 'peg', 'netDebtToPrice', 'dividend_payout_ratio',
       'EVEbitdaRatio', 'EVGP', 'deriv_4m', 'deriv_5m', 'deriv_6m',
       'sma_50d_to_sma_200d_ratio', 'combined_valuation_score',
       'sma10_yoy_growth']

best subset of features so far (for xgboost : ) :

['price', 'est_eps_growth', 'GP', 'netIncome', 'cashcasheq', 'EV',
       'dist_max8M_4M', 'netDebtToPrice', 'dividend_payout_ratio',
       'fwdPriceTosale_diff']

['price', 'est_eps_growth', 'sma_200d', 'GP', 'netIncome', 'marketCap',
       'cashcasheq', 'EV', 'dist_max8M_4M', 'netDebtToPrice',
       'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_6m']
['price', 'est_eps_growth', 'sma_100d', 'sma_200d', 'month', 'GP',
       'netIncome', 'marketCap', 'cashcasheq', 'EV', 'dist_max8M_4M',
       'markRevRatio', 'netDebtToPrice', 'dividend_payout_ratio', 'EVGP',
       'fwdPriceTosale_diff', 'deriv_6m', 'deriv_min8M']

"""
"""
these variables are almost the same (correlation>=0.99): 

"""
MODEL_SELECTED = 'XGBOOST' #test with 'RF' then 'XGBOOST'


df = pd.read_csv('metrics/Metric7/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

df = pd.get_dummies(df, columns=['sector'])

X = df.drop(columns=['to_buy', 'date', 'symbol','min_in_4M','sma_10d_3months_ago','sma_50w'], errors='ignore')
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

for k in range(6,24,2):
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