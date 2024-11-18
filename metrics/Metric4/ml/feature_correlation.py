import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
to run : py -m metrics.Metric4.ml.feature_correlation
"""

"""
best subset of features so far :
['price', 'sma_10d', 'sma_100d', 'sma_200d', 'sma_50d',
       'sma_10d_11months_ago', 'marketCap', 'markRevRAtio', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', '3M_return']
"""
feature_importance = ['price', 'sma_10d', 'sma_100d', 'sma_200d', 'sma_50d',
       'sma_10d_11months_ago', 'marketCap', 'markRevRAtio', 'peg',
       'EVEbitdaRatio', 'price_to_sma_100d_ratio', 'sma_50d_to_sma_100d_ratio',
       'sma_100d_to_sma_200d_ratio', 'value_score', 'return_to_ps']

feature_importance = ['price', 'sma_10d', 'sma_100d', 'sma_200d', 'sma_10d_1weeks_ago',
       'sma_10d_5months_ago', 'sma_10d_7months_ago', 'sma_10d_11months_ago',
       'income_lag_days', 'marketCap', 'EV', 'markRevRatio', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', '3M_return', 'pe_sector_relative',
       '6M_return_sector_relative', '4M_return_sector_relative']

feature_importance = ['price', 'sma_50d', 'sma_10d_1months_ago', 'sma_10d_3months_ago',
       'sma_10d_4months_ago', 'sma_10d_6months_ago', 'marketCap',
       'balance_lag_days', 'markRevRatio', 'pe', 'peg', 'EVEbitdaRatio',
       'EVGP', 'fwdPriceTosale_diff', 'deriv_4m', '3M_return',
       'peg_sector_relative', '3M_return_sector_relative',
       '6M_return_sector_relative', '2M_return_sector_relative']

feature_importance =['price', 'sma_10d_2months_ago', 'income_lag_days', 'marketCap',
       'min_in_4M', 'min_in_8M', 'min8M_lag', 'markRevRatio', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_4m', '3M_return',
       'sma10_yoy_growth', 'peg_sector_relative', '3M_return_sector_relative',
       '6M_return_sector_relative', '2M_return_sector_relative']

feature_importance =['price', 'marketCap', 'balance_lag_days', 'max_in_4M', 'max_minus_min',
       'max_minus_min8M', 'markRevRatio', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP',
       'EVRevenues', 'fwdPriceTosale_diff', 'deriv_4m', '3M_return',
       'sma10_yoy_growth', 'peg_sector_relative', '3M_return_sector_relative',
       '6M_return_sector_relative', '2M_return_sector_relative']

feature_importance =['price', 'marketCap', 'balance_lag_days', 'max_minus_min', 'max_in_8M',
       'min8M_lag', 'max_minus_min8M', 'markRevRatio', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale_diff',
       '3M_return', 'pe_sector_relative', '4M_return_sector_relative',
       '6Mreturn_sector_comp', '2Mreturn_sector_comp', 'peg_sector_comp']

feature_importance =['price', 'income_lag_days', 'marketCap', 'max_minus_min', 'min8M_lag',
       'max_minus_min8M', 'markRevRatio', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP',
       'EVRevenues', 'deriv_3m', 'deriv_4m', '3M_return', 'pe_sector_relative',
       'peg_sector_relative', '6M_return_sector_relative',
       '1M_return_sector_relative', '5M_return_sector_relative']

feature_importance = ['price', 'marketCap', 'balance_lag_days', 'max_in_1M', 'min_in_1M',
       'min8M_lag', 'max_minus_min8M', 'markRevRatio', 'pe', 'peg',
       'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'var_sma50D_100D', 'deriv_5m', '3M_return', 'pe_sector_relative',
       'peg_sector_relative', '1M_return_sector_relative',
       '4M_return_sector_relative', '7M_return_sector_relative',
       '6Mreturn_sector_comp', '2Mreturn_sector_comp']
feature_importance =['price', 'income_lag_days', 'marketCap', 'max_minus_min',
       'max_minus_min1M', 'min8M_lag', 'max_minus_min8M', 'markRevRatio', 'pe',
       'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale',
       'var_sma50D_100D', 'deriv_1m', 'deriv_5m', '3M_return',
       'pe_sector_relative', 'peg_sector_relative',
       '2M_return_sector_relative', '4M_return_sector_relative',
       '7M_return_sector_relative', '6Mreturn_sector_comp']

feature_importance =['price', 'marketCap', 'balance_lag_days', 'max_minus_min',
       'max_minus_min1M', 'max_in_2W', 'min_in_2W', 'min8M_lag',
       'max_minus_min8M', 'markRevRatio', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP',
       'EVRevenues', 'fwdPriceTosale', 'deriv_5m', 'deriv_6m', '3M_return',
       'pe_sector_relative', 'peg_sector_relative',
       'EVEbitdaRatio_sector_relative', '1M_return_sector_relative',
       '2M_return_sector_relative', '4M_return_sector_relative',
       '7M_return_sector_relative', 'deriv_3m_sector_relative',
       '6Mreturn_sector_comp']

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric4/ml/processed_data.csv')

X = df.drop(columns=['symbol', 'date', 'to_buy'])
X= df[feature_importance].drop(columns=[])
y = df['to_buy']

print(X.columns)
correlation_matrix = X.corr().abs()

# Find highly correlated features
threshold = 0.98
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

