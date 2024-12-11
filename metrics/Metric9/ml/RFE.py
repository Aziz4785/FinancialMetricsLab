import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
"""
to run : py -m metrics.Metric9.ml.RFE
"""

"""
best subset of features so far (for random forest : ) :
['price', 'sma_50w', 'sma_100d', 'peg', 'EVRevenues',
       'combined_valuation_score']
['sma_10d', 'sma_10d_4months_ago', 'sma_10d_11months_ago', 'marketCap',
       'markRevRatio', 'EVEbitdaRatio']
['price', 'sma_50w', 'sma_100d', 'sma_50d', 'marketCap', 'peg',
       'EVRevenues', 'deriv_5m', 'combined_valuation_score',
       'sma10_yoy_growth']
['sma_10d', 'sma_10d_4months_ago', 'sma_10d_11months_ago', 'month',
       'marketCap', 'max_minus_min', 'min8M_lag', 'markRevRatio', 'pe',
       'EVEbitdaRatio', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2m',
       'deriv_3m', 'deriv_4m', 'deriv_5m', 'deriv_6m', '7M_return',
       '1y_return', '1Y_6M_growth']
['sma_10d', 'sma_10d_4months_ago', 'sma_10d_11months_ago', 'month',
       'netIncome', 'marketCap', 'max_minus_min', 'min8M_lag', 'markRevRatio',
       'pe', 'EVEbitdaRatio', 'fwdPriceTosale', 'fwdPriceTosale_diff',
       'deriv_2m', 'deriv_3m', 'deriv2_1_3m', 'deriv_4m', 'deriv_5m',
       'deriv_6m', '7M_return', '1y_return', '1Y_6M_growth']

best subset of features so far (for xgboost : ) :
['gp_ratio', 'month', 'GP', 'netIncome', 'marketCap', 'other_expenses',
       'cashcasheq', 'netDebt', 'min_in_4M', 'max_minus_min', 'maxPercen_4M',
       'dist_max8M_4M', 'dist_min8M_4M', 'max_minus_min8M', 'Spread4Mby8M',
       'markRevRatio', 'PS_to_PEG', 'netDebtToPrice', 'fwdPriceTosale_diff',
       'combined_valuation_score']

       ['price', 'gp_ratio', 'month', 'RnD_expenses', 'GP', 'netIncome',
       'total_debt', 'other_expenses', 'future_est_rev', 'min_in_4M',
       'max_minus_min', 'dist_max8M_4M', 'dist_min8M_4M', 'max_minus_min8M',
       'ebitdaMargin', 'Spread4Mby8M', 'peg', 'PS_to_PEG', 'netDebtToPrice',
       'fwdPriceTosale', 'fwdPriceTosale_diff', 'combined_valuation_score']
['gp_ratio', 'sma_10d', 'month', 'RnD_expenses', 'GP', 'netIncome',
       'marketCap', 'total_debt', 'other_expenses', 'cashcasheq',
       'curr_est_eps', 'max_minus_min', 'maxPercen_4M', 'dist_min8M_4M',
       'max_minus_min8M', 'ebitdaMargin', 'Spread4Mby8M', 'markRevRatio',
       'eps_growth', 'pe', 'PS_to_PEG', 'EVEbitdaRatio', 'fwdPriceTosale',
       'fwdPriceTosale_diff', 'deriv_min8M', '1y_return']
"""
"""
these variables are almost the same (correlation>=0.99): 

"""
MODEL_SELECTED = 'XGBOOST' #test with 'RF' then 'XGBOOST'


df = pd.read_csv('metrics/Metric9/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

# columns_to_drop = [col for col in df.columns if col not in my_favourite_features]
# df = df.drop(columns=columns_to_drop)

df = pd.get_dummies(df, columns=['sector'])

#X = df.drop(columns=['to_buy', 'date', 'symbol'], errors='ignore')
X = df.drop(columns=['symbol', 'date', 'to_buy','curr_est_rev','combined_valuation_score','price_to_sma_10d_ratio','sma10_yoy_growth','sma_10d_to_sma_50d_ratio','sma_50d_to_sma_100d_ratio','balance_lag_days','var_sma10D_100D','price_to_SMA10d_1W','5M_return','sma_10d_1weeks_ago','EVRevenues','sector','peg','sma_10d_2weeks_ago',
                     'open_price','sma_10d_5months_ago','price_to_sma_50d_ratio','sma_10d_6months_ago','min_in_8M',
                     'vwap','max_in_1M','sma_10d_7months_ago','min_in_1M','max_in_2W','min_in_2W','EV','EVGP',
                     'sma_50d','sma_10d_1months_ago','sma_10d_2months_ago','sma_10d_3months_ago','price',
                     'sma_100d','max_in_8M','sma_50w','min_in_4M','max_in_4M','future_est_rev','price_to_sma_100d_ratio'], errors='ignore')
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

for k in range(6,30,2):
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