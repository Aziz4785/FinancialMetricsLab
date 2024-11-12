import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

"""
to run : py -m experiment1.rf_feature_importance
"""

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data.csv')
# df = df[df['dividend_payout_ratio'] >= -0.02]
# df = df[df['est_rev_growth'] >= 0.09]
# df = df[df['dividend_payout_ratio'] <= 0.1]
df = df.sample(frac=1).reset_index(drop=True)
# Remove 'date' and 'symbol' columns

X = df.drop(columns=['target', 'date', 'stock'])
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train the model
rf_model = RandomForestClassifier(n_estimators=400,random_state=42)
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

sorted_features = feature_importances.index.tolist()
print("\nList of features sorted by importance:")
print(sorted_features)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title("Feature Importances in Random Forest Model")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()