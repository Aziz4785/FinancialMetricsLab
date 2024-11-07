import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .utils import *

"""
to run : py -m experiment1.experiment2
"""
SHOW_FEATURE_IMPORTANCE = False
DEPTH = 2

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data.csv')
MIN_SAMPLE_LEAFS = int(0.1*len(df))

# Prepare features (X) and target (y)
exclude_columns = ['target', 'stock', 'date']
feature_columns = [col for col in df.columns if col not in exclude_columns]

X = df[feature_columns]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree with max_depth=1
dt = DecisionTreeClassifier(max_depth=DEPTH,min_samples_leaf=MIN_SAMPLE_LEAFS)
dt.fit(X_train, y_train)

# Calculate and print accuracy
train_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")

# Visualize the tree
# plt.figure(figsize=(20,10))
# plot_tree(dt, 
#           feature_names=feature_columns,
#           class_names=['0', '1'],
#           filled=True,
#           rounded=True,
#           fontsize=12)
# plt.show()

visualize_tree(dt,feature_columns)

if SHOW_FEATURE_IMPORTANCE:
    # Print the feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': dt.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
