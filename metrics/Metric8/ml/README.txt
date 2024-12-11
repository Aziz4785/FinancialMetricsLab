========== Feature Subset 1: ['price', 'sma_10d_11months_ago', 'month', 'netIncome', 'total_debt', 'cashcasheq', 'netDebt', 'curr_est_eps', 'EV', 'max_minus_min', 'dist_max8M_4M', 'dist_min8M_4M', 'ebitdaMargin', 'Spread4Mby8M', 'markRevRatio', 'pe', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'var_sma50D', 'deriv_min8M', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score', 'sector_energy'] ==========
RF4c Results:
specificity = 0.9166666666666666
Precision Score: 0.9225
Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.92      0.90       252
           1       0.92      0.89      0.91       280

    accuracy                           0.90       532
   macro avg       0.90      0.90      0.90       532
weighted avg       0.90      0.90      0.90       532

========== Feature Subset 10: ['sma_200d', 'sma_10d_5months_ago', 'sma_10d_11months_ago', 'income_lag_days', 'marketCap', 'EV', 'max_minus_min', 'min8M_lag', 'Spread4Mby8M', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m', 'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m', 'combined_valuation_score', 'sma10_yoy_growth'] ==========
RF3 Results:
specificity = 0.9047619047619048
Precision Score: 0.9155
Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.90      0.91       252
           1       0.92      0.93      0.92       280

    accuracy                           0.92       532
   macro avg       0.92      0.92      0.92       532
weighted avg       0.92      0.92      0.92       532

XGB5 Results:
specificity = 0.9047619047619048
Precision Score: 0.9161
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.90      0.92       252
           1       0.92      0.94      0.93       280

    accuracy                           0.92       532
   macro avg       0.92      0.92      0.92       532
weighted avg       0.92      0.92      0.92       532


========== Feature Subset 4: ['sma_10d_6months_ago', 'marketCap', 'balance_lag_days', 'max_minus_min', 'min8M_lag', 'Spread4Mby8M', 'markRevRatio', 'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m', 'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m', '1y_return', 'combined_valuation_score'] ==========


RF2 Results:
specificity = 0.9087301587301587
Precision Score: 0.9193
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.91      0.92       252
           1       0.92      0.94      0.93       280

    accuracy                           0.92       532
   macro avg       0.92      0.92      0.92       532
weighted avg       0.92      0.92      0.92       532


RF4c Results:
specificity = 0.9126984126984127
Precision Score: 0.9220
Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.91      0.92       252
           1       0.92      0.93      0.93       280

    accuracy                           0.92       532
   macro avg       0.92      0.92      0.92       532
weighted avg       0.92      0.92      0.92       532


XGB5 Results:
specificity = 0.9087301587301587
Precision Score: 0.9193
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.91      0.92       252
           1       0.92      0.94      0.93       280

    accuracy                           0.92       532
   macro avg       0.92      0.92      0.92       532
weighted avg       0.92      0.92      0.92       532

========== Feature Subset 3: ['est_eps_growth', 'month', 'GP', 'netIncome', 'ebitda', 'marketCap', 'cashcasheq', 'curr_est_eps', 'max_minus_min', 'dist_max8M_4M', 'dist_min8M_4M', 'ebitdaMargin', 'Spread4Mby8M', 'markRevRatio', 'peg', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_min8M', '1y_return', '3M_2M_growth', 'price_to_sma_100d_ratio', 'sma_10d_to_sma_50d_ratio', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score'] ==========
RF1 Results:
specificity = 0.9206349206349206
Precision Score: 0.9267
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.92      0.91       252
           1       0.93      0.90      0.92       280

    accuracy                           0.91       532
   macro avg       0.91      0.91      0.91       532
weighted avg       0.91      0.91      0.91       532

========== Feature Subset 11: ['sma_10d_5months_ago', 'month', 'ebitda', 'cashcasheq', 'netDebt', 'curr_est_eps', 'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M', 'Spread4Mby8M', 'debtToPrice', 'peg', 'PS_to_PEG', 'dividend_payout_ratio', 'EVEbitdaRatio', 'fwdPriceTosale_diff', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score'] ==========
RF Results:
specificity = 0.9206349206349206
Precision Score: 0.9254
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.92      0.90       252
           1       0.93      0.89      0.91       280

    accuracy                           0.90       532
   macro avg       0.90      0.90      0.90       532
weighted avg       0.90      0.90      0.90       532