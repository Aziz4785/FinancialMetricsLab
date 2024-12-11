import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
to run : py -m metrics.Metric9.ml.feature_correlation
"""
threshold = 0.96

feature_importance = ['sma_200d', 'sma_50d', 'sma_10d_5months_ago', 'sma_10d_11months_ago',
'income_lag_days', 'marketCap', 'EV', 'max_minus_min', 'markRevRatio',
'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'deriv_1m',
'deriv_2m', 'deriv_4m', 'deriv_5m', 'combined_valuation_score',
'sma10_yoy_growth']

df = pd.read_csv('metrics/Metric9/ml/processed_data.csv')

X = df.drop(columns=['symbol', 'date', 'to_buy','curr_est_rev','combined_valuation_score','price_to_sma_10d_ratio','sma10_yoy_growth','sma_10d_to_sma_50d_ratio','sma_50d_to_sma_100d_ratio','balance_lag_days','var_sma10D_100D','price_to_SMA10d_1W','5M_return','sma_10d_1weeks_ago','EVRevenues','sector','peg','sma_10d_2weeks_ago',
                     'open_price','sma_10d_5months_ago','price_to_sma_50d_ratio','sma_10d_6months_ago','min_in_8M',
                     'vwap','max_in_1M','sma_10d_7months_ago','min_in_1M','max_in_2W','min_in_2W','EV','EVGP',
                     'sma_50d','sma_10d_1months_ago','sma_10d_2months_ago','sma_10d_3months_ago','price',
                     'sma_100d','max_in_8M','sma_50w','min_in_4M','max_in_4M','future_est_rev','price_to_sma_100d_ratio'])
#X= df[feature_importance].drop(columns=[])
y = df['to_buy']

print(X.columns)
# Print data types of each column
print("Column datatypes:")
print(X.dtypes)

# Identify numeric and non-numeric columns
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
non_numeric_columns = X.select_dtypes(exclude=['int64', 'float64']).columns

print("\nNon-numeric columns:", list(non_numeric_columns))


correlation_matrix = X.corr().abs()

# Find highly correlated features

high_corr_features = np.where(correlation_matrix > threshold)
high_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                      for x, y in zip(*high_corr_features) if x != y and x < y]

# low_corr_features = np.where(correlation_matrix < 0.1)
# low_corr_features = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
#                       for x, y in zip(*low_corr_features) if x != y and x < y]

    


print("Highly correlated feature pairs:")
feature_by_appearance = {}
feature_by_avg_corr_score = {}
for feat1, feat2, corr in high_corr_features:
    print(f"{feat1} - {feat2}: {corr:.2f}")
    feature_by_appearance[feat1] = feature_by_appearance.get(feat1, 0) + 1
    feature_by_appearance[feat2] = feature_by_appearance.get(feat2, 0) + 1
    feature_by_avg_corr_score[feat1] = feature_by_avg_corr_score.get(feat1, 0) + corr
    feature_by_avg_corr_score[feat2] = feature_by_avg_corr_score.get(feat2, 0) + corr


print()
print()
feature_by_appearance = dict(sorted(feature_by_appearance.items(), key=lambda item: item[1]))
feature_by_avg_corr_score = dict(sorted(feature_by_avg_corr_score.items(), key=lambda item: item[1]))
for key in feature_by_avg_corr_score:
    feature_by_avg_corr_score[key]=float(feature_by_avg_corr_score[key]/feature_by_appearance[key])



for key, value in feature_by_appearance.items():
    if value>=0:
        position = feature_importance.index(key) if key in feature_importance else -1
        print(f"'{key}': {value}   -> position : {position} -> avg_score : {feature_by_avg_corr_score[key]}")

print()
print()

