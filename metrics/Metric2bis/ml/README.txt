Feature Subset 5: ['revenues', 'price', 'pe', 'ps_ratio', 'sma_100d', 'sma_200d']
RF4a Results:
specificity = 0.9786184210526315
Precision Score: 0.9774
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       608
           1       0.98      0.97      0.97       582

    accuracy                           0.97      1190
   macro avg       0.97      0.97      0.97      1190
weighted avg       0.97      0.97      0.97      1190



Feature Subset 4: ['price', 'sma_100d', 'sma_50w', 'sma_200d', 'sma_50d', 'GP', 'dividendsPaid', 'marketCap', 'EV', 'max_in_1M', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale', 'price_to_sma_200d_ratio', 'combined_valuation_score']
RF4a Results:
specificity = 0.975328947368421
Precision Score: 0.9743
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       608
           1       0.97      0.98      0.98       582

    accuracy                           0.98      1190
   macro avg       0.98      0.98      0.98      1190
weighted avg       0.98      0.98      0.98      1190


XGB4 Results:
specificity = 0.9736842105263158
Precision Score: 0.9727
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.97      0.98       608
           1       0.97      0.98      0.98       582

    accuracy                           0.98      1190
   macro avg       0.98      0.98      0.98      1190
weighted avg       0.98      0.98      0.98      1190