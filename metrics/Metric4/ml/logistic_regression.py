from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
to run : py -m metrics.Metric4.ml.logistic_regression
"""
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric4/ml/processed_data.csv')
df = df.sample(frac=1).reset_index(drop=True)
# Removing 'date' and 'symbol' columns
df = df.drop(['date', 'symbol'], axis=1)

# Separating features and target
selected_features = ['divpayout_ratio', 'rev_growth', 'income_lag_days',
       'max_minus_min', 'debtToPrice', 'markRevRAtio',
       'eps_growth', 'pe', 'peg', 'EVEbitdaRatio', '1Y_6M_growth']
X = df[selected_features]
y = df['to_buy']


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initializing and fitting the model
model = LogisticRegression()  # Adjusting max_iter for potential convergence
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)

yTRAIN_pred = model.predict(X_train_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_output = classification_report(y_test, y_pred)

print(f"accuracy : {accuracy}")
print(classification_report_output)

training_accuracy = accuracy_score(y_train, yTRAIN_pred)

print(f"training_accuracy : {training_accuracy}")