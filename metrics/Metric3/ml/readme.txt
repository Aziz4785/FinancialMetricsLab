========== Feature Subset 4: ['price', 'MarkRevRatio', 'sma_200d', 'ratio', 'fcf', 'sma_50w'] ==========
----------------------

Random Forest Results:
specificity = 0.9830076465590484
Precision Score: 0.9836
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1177
           1       0.98      0.98      0.98      1217

    accuracy                           0.98      2394
   macro avg       0.98      0.98      0.98      2394
weighted avg       0.98      0.98      0.98      2394


========== Feature Subset 12: ['MarkRevRatio', 'ratio', 'sma_200d', 'peRatio', 'sma_50w', 'sma_100d'] ========== 
----------------------

Random Forest 200 Results:
specificity = 0.9830076465590484
Precision Score: 0.9836
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1177
           1       0.98      0.98      0.98      1217

    accuracy                           0.98      2394
   macro avg       0.98      0.98      0.98      2394
weighted avg       0.98      0.98      0.98      2394

========== Feature Subset 14: ['MarkRevRatio', 'ratio', 'sma_200d', 'peRatio', 'sma_50w', 'sma_100d', 'price', 'other_expenses'] ==========
----------------------

XGBoost c4 Results:
specificity = 0.983857264231096
Precision Score: 0.9842
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98      1177
           1       0.98      0.97      0.98      1217

    accuracy                           0.98      2394
   macro avg       0.98      0.98      0.98      2394
weighted avg       0.98      0.98      0.98      2394

========== Feature Subset 21: ['peRatio', 'ratio', 'MarkRevRatio', 'price', 'sma_200d', 'sma_50w', 'fcf', 'sma_100d', 'ocfToNetinc', 'price_to_50w'] ==========
XGBoost 200 Results:
specificity = 0.9821580288870009
Precision Score: 0.9827
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98      1177
           1       0.98      0.98      0.98      1217

    accuracy                           0.98      2394
   macro avg       0.98      0.98      0.98      2394
weighted avg       0.98      0.98      0.98      2394