import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

"""
to run : py -m experiment1.experiment
"""

def analyze_feature_importance(df, target_col='target'):
    """
    Analyze feature importance using multiple methods
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    target_col (str): Name of target column
    
    Returns:
    dict: Dictionary containing results from different methods
    """
    # Prepare the data
    # Remove date and stock columns as they're not numerical features
    feature_cols = [col for col in df.columns if col not in [target_col, 'date', 'stock']]
    X = df[feature_cols]
    y = df[target_col]
    
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X, y)
    mi_results = dict(zip(feature_cols, mi_scores))
    
    # 2. ANOVA F-test
    f_scores, _ = f_classif(X, y)
    f_results = dict(zip(feature_cols, f_scores))
    
    # 3. Decision Tree Feature Importance
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X, y)
    dt_results = dict(zip(feature_cols, dt.feature_importances_))
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    

    mi_df = pd.DataFrame.from_dict(mi_results, orient='index', columns=['Score'])
    mi_df.sort_values('Score', ascending=True).plot(kind='barh')
    print("Mutual Information results: ")
    print(mi_df)

    
    # Plot 2: ANOVA F-scores
    print("anova f score : ")
    f_df = pd.DataFrame.from_dict(f_results, orient='index', columns=['Score'])
    f_df.sort_values('Score', ascending=True).plot(kind='barh')
    print(f_df)
    
    # Plot 3: Decision Tree Feature Importance
    plt.subplot(3, 1, 3)
    dt_df = pd.DataFrame.from_dict(dt_results, orient='index', columns=['Score'])
    dt_df.sort_values('Score', ascending=True).plot(kind='barh')
    plt.title('Decision Tree Feature Importance')
    plt.xlabel('Score')
    
    plt.tight_layout()
    plt.show()  # Show the first figure with feature importance plots
    
    # Box plots for top features
    # Get top 3 features based on mutual information
    top_features = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)[:3]
    top_feature_names = [f[0] for f in top_features]
    
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(top_feature_names, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x=target_col, y=feature, data=df)
        plt.title(f'Distribution of {feature} by Target')
    
    plt.tight_layout()
    plt.show()  # Show the second figure with box plots
    
    return {
        'mutual_info': mi_results,
        'f_scores': f_results,
        'decision_tree': dt_results,
        'top_features': top_features
    }

def calculate_gini_split(data: pd.DataFrame, feature: str, target: str = 'target') -> Tuple[float, float]:
    """
    Calculate the Gini impurity for a binary split on a numerical feature.
    Returns the best split point and corresponding Gini impurity.
    """
    # Sort the data by the feature
    sorted_data = data.sort_values(by=feature)
    
    # Get unique values for potential splits
    split_points = sorted_data[feature].unique()[:-1]  # exclude last point
    
    best_gini = float('inf')
    best_split = None
    
    total_samples = len(data)
    
    for split in split_points:
        # Split the data
        left_mask = data[feature] <= split
        right_mask = ~left_mask
        
        # Calculate proportions
        left_prop = sum(left_mask) / total_samples
        right_prop = sum(right_mask) / total_samples
        
        # Calculate Gini impurity for each split
        left_target = data[left_mask][target]
        right_target = data[right_mask][target]
        
        left_gini = 1 - np.sum([(sum(left_target == c) / len(left_target))**2 
                               for c in [0, 1]]) if len(left_target) > 0 else 0
        right_gini = 1 - np.sum([(sum(right_target == c) / len(right_target))**2 
                                for c in [0, 1]]) if len(right_target) > 0 else 0
        
        # Calculate weighted Gini impurity
        weighted_gini = left_prop * left_gini + right_prop * right_gini
        
        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_split = split
            
    return best_split, best_gini


def find_best_feature(data: pd.DataFrame, target: str = 'target') -> dict:
    """
    Analyze all numerical features to find the one that gives the best split
    based on Gini impurity.
    """
    print("select the minimum gini impurity")
    # Exclude date and target columns
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features = [f for f in numerical_features if f != target]
    
    results = {}
    for feature in numerical_features:
        split_point, gini = calculate_gini_split(data, feature, target)
        results[feature] = {
            'gini_impurity': gini,
            'split_point': split_point
        }
    
    # Sort results by Gini impurity
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['gini_impurity']))
    return sorted_results


# Example usage:
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data.csv')
#results = analyze_feature_importance(df)
results = find_best_feature(df)
print("\nFeature Analysis Results:")
for feature, metrics in results.items():
    print(f"\nFeature: {feature}")
    print(f"Gini Impurity: {metrics['gini_impurity']:.4f}")
    print(f"Best Split Point: {metrics['split_point']:.4f}")
