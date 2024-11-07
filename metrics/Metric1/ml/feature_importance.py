import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

"""
to run : py -m metrics.Metric1.ml.feature_importance
"""

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
# Remove 'date' and 'symbol' columns
df = df.drop(['date', 'symbol'], axis=1)

# Assuming df is your DataFrame
columns_to_use =['price','ratio','sma_10d','sma_50w','sma_100d','sma_200d','other_expenses','NetProfitMargin','ebitdaMargin','eps','costOfRevenue','netIncome','revenues','revenues_lag_days','MarkRevRatio','cost_revenue_ratio']
available_columns = [col for col in columns_to_use if col in df.columns]

X = df[available_columns]
y = df['to_buy']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train the model
rf_model = RandomForestClassifier(n_estimators=500,random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Get feature importances
importances = rf_model.feature_importances_
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

# Print feature importances
print("Feature Importances:")
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title("Feature Importances in Random Forest Model")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()