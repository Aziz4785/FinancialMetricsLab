import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
to run : py -m metrics.Metric5.ml.feature_correlation
"""


feature_importance = ['price', 'evebitda', 'sma_50d', 'marketCap', 'EV', 'min_in_1M',
       'max_in_2W', 'debtToPrice', 'markRevRatio', 'pe', 'peg', 'EVGP',
       'EVRevenues', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'var_sma100D',
       'var_sma50D_100D', 'var_sma10D_100D', 'deriv_1m', 'deriv_2m',
       '4M_return', '1y_return', '2M_1W_growth', '1Y_6M_growth',
       'price_to_sma_100d_ratio', 'peg_sector_relative',
       'markRevRatio_sector_relative', 'deriv_4m_sector_relative']

feature_importance =['var_sma50D_100D', 'future_est_rev',
       '1y_return', 'debtToPrice', '1Y_6M_growth', 'EVGP',
        'marketCap', 'var_sma10D_50D', 'var_sma100D',
       'max_minus_min8M', 'curr_est_eps', 'fwdPriceTosale',
       'combined_valuation_score', 'markRevRatio',
        'deriv_4m', 'sma_10d',
       'evebitda', 'EV', 'dividend_payout_ratio', 'peg', 'netIncome']

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/ml/processed_data.csv')

X = df.drop(columns=['symbol', 'date', 'to_buy'])
X= df[feature_importance].drop(columns=[])
y = df['to_buy']

print(X.columns)
correlation_matrix = X.corr().abs()

# Find highly correlated features
threshold = 0.85
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

