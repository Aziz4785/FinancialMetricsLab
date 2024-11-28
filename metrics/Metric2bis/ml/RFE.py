import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
"""
to run : py -m metrics.Metric2bis.ml.RFE
"""
"""
best subset of features so far (for random forest : ) :
['price', 'sma_200d', 'GP', 'marketCap', 'EV', 'max_in_4M', 'max_in_2W',
       'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues']

['price', 'sma_200d', 'peg', 'EVEbitdaRatio']

['price', 'sma_100d', 'sma_200d', 'sma_50d', 'GP',
       'dividendsPaid', 'marketCap', 'EV', 'max_in_1M', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'price_to_sma_200d_ratio', 'combined_valuation_score']

best subset of features so far (for xgboost : ) :
['dividendsPaid', 'ebitda', 'total_debt', 'dividend_payout_ratio']

['month', 'revenues', 'GP', 'dividendsPaid', 'ebitda', 'total_debt',
       'cashcasheq', 'netDebt', 'future_est_rev', 'EV', 'max_in_4M',
       'eps_growth', 'peg', 'dividend_payout_ratio', 'fwdPriceTosale',
       'fwdPriceTosale_diff']
"""
"""
these variables are almost the same (correlation>=0.99): 
price - sma_100d - sma_50d - max_in_1M
sma_50w - sma_200d
peg - combined_valuation_score: 1.00
"""
MODEL_SELECTED = 'XGBOOST' #'RF' or 'XGBOOST'


df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric2bis/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

df = pd.get_dummies(df, columns=['sector'])

X = df.drop(columns=['to_buy', 'date', 'symbol','sma_100d','sma_50d','max_in_1M','combined_valuation_score','sma_50w'], errors='ignore')
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

for k in range(4,22,2):
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