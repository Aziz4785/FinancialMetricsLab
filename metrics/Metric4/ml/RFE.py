import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
"""
to run : py -m metrics.Metric4.ml.RFE
"""
"""
best subset of features so far :
['price', 'sma_10d', 'sma_100d', 'sma_200d', 'sma_50d',
       'sma_10d_11months_ago', 'marketCap', 'markRevRAtio', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', '3M_return']

['price', 'sma_10d', 'sma_100d', 'sma_200d', 'sma_50d',
       'sma_10d_11months_ago', 'marketCap', 'markRevRAtio', 'peg',
       'EVEbitdaRatio', 'price_to_sma_100d_ratio', 'sma_50d_to_sma_100d_ratio',
       'sma_100d_to_sma_200d_ratio', 'value_score', 'return_to_ps'],

['price', 'sma_10d', 'sma_100d', 'sma_200d', 'sma_50d',
'sma_10d_1weeks_ago', 'sma_10d_5months_ago', 'sma_10d_11months_ago',
'marketCap', 'balance_lag_days', 'EV', 'markRevRatio', 'pe', 'peg',
'EVEbitdaRatio', 'EVGP', '3M_return', '3M_return_sector_relative']

['price', 'income_lag_days', 'marketCap', 'max_minus_min', 'min8M_lag',
       'max_minus_min8M', 'markRevRatio', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP',
       'EVRevenues', 'deriv_3m', 'deriv_4m', '3M_return', 'pe_sector_relative',
       'peg_sector_relative', '6M_return_sector_relative',
       '1M_return_sector_relative', '5M_return_sector_relative']
"""
"""
these variables are almost the same (correlation>=0.99): 
-price sma_10d max_in_4M max_in_8M min_in_4M min_in_8M sma_10d_2months_ago sma_50d sma_100d sma_10d_1weeks_ago sma_200d sma_10d_5months_ago sma_10d_7months_ago sma_10d_11months_ago
-marketCap - EV
-markRevRatio - fwdPriceTosale_diff
-peg - peg_sector_comp
"""


df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric4/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

df = pd.get_dummies(df, columns=['sector'])

X = df.drop(columns=['to_buy', 'date', 'symbol','peg_sector_comp','fwdPriceTosale_diff','max_in_8M','max_in_4M','sma_10d_2months_ago','min_in_8M','min_in_4M','EV','sma_50d','sma_10d_6months_ago','sma_10d_3months_ago','sma_10d_4months_ago','sma_10d_1months_ago','sma_10d' ,'sma_100d', 'sma_10d_1weeks_ago' ,'sma_200d', 'sma_10d_5months_ago', 'sma_10d_7months_ago', 'sma_10d_11months_ago'], errors='ignore')
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
rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

for k in range(6,24,2):
    print(f"-----number of features : {k}")
    # Initialize RFE with the RF model and specify the number of features to select
    # You can choose 'n_features_to_select' based on your requirements
    # For example, to select the top 10 features:
    RFE_selector = RFE(estimator=rf_model, n_features_to_select=k, step=1)
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

    # Train the RandomForest model on the selected features
    rf_model.fit(X_train_selected, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test_selected)

    # Evaluate the model
    print("\nModel Performance with RFE-selected Features:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))