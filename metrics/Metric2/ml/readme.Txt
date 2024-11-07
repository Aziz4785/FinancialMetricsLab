========== Feature Subset 10: ['revenues', 'price', 'pe_ratio', 'ebitdaMargin', 'ratio', 'sma_100d', 'sma_200d'] ==========  

----------------------

Random Forest 200 Results:
specificity = 0.9826989619377162
Precision Score: 0.9842
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.98      0.99       289
           1       0.98      0.99      0.99       314

    accuracy                           0.99       603
   macro avg       0.99      0.99      0.99       603
weighted avg       0.99      0.99      0.99       603

----------------------

XGBoost Results:
specificity = 0.9792387543252595
Precision Score: 0.9809
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       289
           1       0.98      0.98      0.98       314

    accuracy                           0.98       603
   macro avg       0.98      0.98      0.98       603
weighted avg       0.98      0.98      0.98       603