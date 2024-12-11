import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from xgboost import XGBClassifier
"""
to run : py -m metrics.Metric8.ml.RFE
"""
my_favourite_features = ['to_buy','est_eps_growth','month','open_price','sector','income_lag_days','balance_lag_days','curr_est_eps','max4M_lag','min4M_lag','max_minus_min','var_price_to_minmax','maxPercen_4M','max1M_lag','min1M_lag','max_minus_min1M','var_price_to_minmax1M','maxPercen_1M','max2W_lag','min2W_lag','max_minus_min2W','var_price_to_minmax2W','maxPercen_2W','max8M_lag','dist_max8M_4M','min8M_lag','dist_min8M_4M','max_minus_min8M','var_price_minmax8M','maxPercen_8M','ebitdaMargin','Spread1Mby8M','Spread4Mby8M','Spread1Mby4M','debtToPrice','markRevRatio','open_price_to_now','eps_growth','pe','peg','PS_to_PEG','netDebtToPrice','dividend_payout_ratio','EVEbitdaRatio','fwdPriceTosale_diff','var_sma50D','var_sma100D','var_sma10D','var_sma10D_50D','var_sma50D_100D','var_sma50D_200D','var_sma10D_100D','var_sma10D_200D','relative_std','deriv_1w','deriv_1m','deriv_2m','deriv_3m','deriv_4m','deriv_5m','deriv_6m','deriv_min4M','deriv_max8M','deriv_min8M','deriv_1w1m_growth','deriv_1m2m_growth','deriv_3m4m_growth','1W_return','price_to_SMA10d_1W','price_to_SMA10d_5M','1M_return','2M_return','3M_return','4M_return','5M_return','6M_return','7M_return','1y_return','1M_1W_growth','2M_1W_growth','1M1W_2M1W_growth','2M_1M_growth','3M_2M_growth','4M_3M_growth','1Y_6M_growth','price_to_sma_10d_ratio','price_to_sma_50d_ratio','price_to_sma_100d_ratio','price_to_sma_200d_ratio','sma_10d_to_sma_50d_ratio','sma_10d_to_sma_100d_ratio','sma_10d_to_sma_200d_ratio','sma_50d_to_sma_100d_ratio','sma_50d_to_sma_200d_ratio','sma_100d_to_sma_200d_ratio','golden_cross','death_cross','smas_above_count','price_above_all_sma','price_below_all_sma','combined_valuation_score','market_cap_quintile','return_to_pe','sma10_yoy_growth']

"""
best subset of features so far (for random forest : ) :
['sma_200d', 'EV', 'pe', 'peg', 'EVGP', 'deriv_2m', 'deriv_4m',
       'combined_valuation_score']

['sma_200d', 'sma_10d_5months_ago', 'marketCap', 'pe', 'EVGP',
       'deriv_2m', 'combined_valuation_score', 'sma10_yoy_growth']

       ['sma_200d', 'sma_50d', 'sma_10d_5months_ago', 'marketCap',
       'max_minus_min', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'deriv_1m',
       'deriv_2m', 'deriv_4m', 'combined_valuation_score', 'sma10_yoy_growth']

['sma_200d', 'sma_50d', 'sma_10d_5months_ago', 'sma_10d_11months_ago',
'income_lag_days', 'marketCap', 'EV', 'max_minus_min', 'markRevRatio',
'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'deriv_1m',
'deriv_2m', 'deriv_4m', 'deriv_5m', 'combined_valuation_score',
'sma10_yoy_growth']

best subset of features so far (for xgboost : ) :
['month', 'ebitda', 'cashcasheq', 'netDebt', 'curr_est_eps',
       'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M', 'debtToPrice',
       'PS_to_PEG', 'dividend_payout_ratio', 'fwdPriceTosale_diff']

['month', 'ebitda', 'cashcasheq', 'netDebt', 'curr_est_eps',
       'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M', 'Spread4Mby8M',
       'debtToPrice', 'PS_to_PEG', 'dividend_payout_ratio', 'EVEbitdaRatio',
       'fwdPriceTosale_diff', 'sma_100d_to_sma_200d_ratio',
       'combined_valuation_score']
       
['sma_50w', 'sma_10d_1months_ago', 'sma_10d_5months_ago', 'month',
       'revenues', 'GP', 'netIncome', 'cashcasheq', 'netDebt', 'curr_est_eps',
       'max_minus_min', 'dist_min8M_4M', 'Spread4Mby8M', 'markRevRatio', 'peg',
       'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff',
       'var_sma50D_100D', 'var_sma50D_200D', 'deriv_min4M', '2M_1W_growth']

"""
"""
these variables are almost the same (correlation>=0.99): 
sma_200d - sma_10d_5months_ago: 0.99
marketCap - EV: 1.00
"""
MODEL_SELECTED = 'RF' #test with 'RF' then 'XGBOOST'


df = pd.read_csv('metrics/Metric8/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

# columns_to_drop = [col for col in df.columns if col not in my_favourite_features]
# df = df.drop(columns=columns_to_drop)

df = pd.get_dummies(df, columns=['sector'])

#X = df.drop(columns=['to_buy', 'date', 'symbol'], errors='ignore')
X = df.drop(columns=['symbol', 'date', 'to_buy','sector','max_in_8M','sma_10d_7months_ago','var_sma10D_50D','total_debt',
                     'max_in_1M','min_in_8M','sma_50w','min_in_1M','min_in_4M','max_in_2W','var_sma50D','5M_return',
                     'sma_10d_4months_ago','vwap','sma_10d_1months_ago','var_sma50D_100D','sma_50d_to_sma_200d_ratio',
                     'sma_10d_2weeks_ago','income_lag_days','price_to_sma_10d_ratio','sma_50d_to_sma_100d_ratio','price',
                     'max_in_4M','sma_10d_1weeks_ago','sma_200d','sma10_yoy_growth','price_to_sma_200d_ratio','std_10d',
                     'min_in_2W','sma_100d','1W_return','var_sma100D','revenues','EVRevenues','pe','sma_10d_11months_ago',
                     'curr_est_rev','sma_10d','sma_50d','eps_growth','death_cross','sma_10d_5months_ago','var_sma10D_100D',
                     'sma_10d_3months_ago','open_price','sma_10d_2months_ago','EV','var_sma10D_200D'], errors='ignore')
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

for k in range(16,30,2):
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