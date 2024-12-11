========== Feature Subset 7: ['price', 'est_eps_growth', 'sma_100d', 'sma_200d', 'month', 'GP', 'netIncome', 'marketCap', 'cashcasheq', 'EV', 'dist_max8M_4M', 'markRevRatio', 'netDebtToPrice', 'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_6m', 'deriv_min8M'] ==========
RF4c Results:
specificity = 0.9819277108433735
Precision Score: 0.9837
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.98      0.97       166
           1       0.98      0.96      0.97       189

    accuracy                           0.97       355
   macro avg       0.97      0.97      0.97       355
weighted avg       0.97      0.97      0.97       355

========== Feature Subset 4: ['price', 'sma_100d', 'sma_200d', 'sma_50d', 'sma_10d_4months_ago', 'sma_10d_6months_ago', 'marketCap', 'EV', 'max_minus_min8M', 'debtToPrice', 'pe', 'peg', 'netDebtToPrice', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'deriv_4m', 'deriv_5m', 'deriv_6m', 'sma_50d_to_sma_200d_ratio', 'combined_valuation_score', 'sma10_yoy_growth'] ==========
RF Results:
specificity = 0.9879518072289156
Precision Score: 0.9891
Classification Report:
              precision    recall  f1-score   support

           0       0.96      0.99      0.97       166
           1       0.99      0.96      0.98       189

    accuracy                           0.97       355
   macro avg       0.97      0.98      0.97       355
weighted avg       0.98      0.97      0.97       355

XGB8 Results:
specificity = 0.9819277108433735
Precision Score: 0.9837
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.98      0.97       166
           1       0.98      0.96      0.97       189

    accuracy                           0.97       355
   macro avg       0.97      0.97      0.97       355
weighted avg       0.97      0.97      0.97       355
