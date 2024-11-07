========== Feature Subset 4: ['MarkRevRatio', 'sma_50w', 'sma_200d', 'sma_100d', 'price', 'sma_10d'] ==========
Random Forest 200 Results:
specificity = 0.9835390946502057
Precision Score: 0.9856
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98       243
           1       0.99      0.97      0.98       282

    accuracy                           0.98       525
   macro avg       0.98      0.98      0.98       525
weighted avg       0.98      0.98      0.98       525

========== Feature Subset 8: ['MarkRevRatio', 'sma_200d', 'price', 'sma_50w'] ==========
XGBoost 200 Results:
specificity = 0.9835390946502057
Precision Score: 0.9857
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98       243
           1       0.99      0.98      0.98       282

    accuracy                           0.98       525
   macro avg       0.98      0.98      0.98       525
weighted avg       0.98      0.98      0.98       525

========== Feature Subset 10: ['MarkRevRatio', 'sma_200d', 'price', 'sma_50w', 'sma_100d', 'ratio'] ==========
----------------------

XGBoost Results:
specificity = 0.9835390946502057
Precision Score: 0.9857
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98       243
           1       0.99      0.98      0.98       282

    accuracy                           0.98       525
   macro avg       0.98      0.98      0.98       525
weighted avg       0.98      0.98      0.98       525


========== Feature Subset 11: ['MarkRevRatio', 'sma_200d', 'price', 'sma_50w', 'sma_100d', 'ratio', 'ebitdaMargin'] ==========
XGBoost Results:
specificity = 0.9876543209876543
Precision Score: 0.9893
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       243
           1       0.99      0.98      0.99       282

    accuracy                           0.98       525
   macro avg       0.98      0.98      0.98       525
weighted avg       0.98      0.98      0.98       525