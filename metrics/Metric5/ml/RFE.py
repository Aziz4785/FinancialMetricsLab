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
best subset of features so far :

"""
"""
these variables are almost the same (correlation>=0.99): 
-peg - peg_sector_comp
-var_sma50D_100D - sma_50d_to_sma_100d_ratio
-var_sma10D_100D - sma_10d_to_sma_100d_ratio
-1y_return - sma10_yoy_growth
-price - min_in_1M  
-price - max_in_2W- sma_10d
-var_sma100D price_to_sma_100d_ratio
"""
MODEL_SELECTED = 'XGBOOST' #'RF' or 'XGBOOST'


df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)

df = pd.get_dummies(df, columns=['sector'])

X = df.drop(columns=['to_buy', 'date', 'symbol','price' ,'max_in_2W', 'peg_sector_comp','sma_50d_to_sma_100d_ratio','sma_10d_to_sma_100d_ratio','sma10_yoy_growth','price_to_sma_100d_ratio'], errors='ignore')
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