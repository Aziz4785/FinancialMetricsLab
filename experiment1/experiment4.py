import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .utils import *

"""
to run : py -m experiment1.experiment4
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class HighDensityClusterClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, min_cluster_size=500, min_purity=0.9, max_depth=5):
        self.min_cluster_size = min_cluster_size
        self.min_purity = min_purity
        self.max_depth = max_depth
        self.tree = DecisionTreeClassifier(max_depth=max_depth)
        self.best_path = None
        self.best_score = 0
        self.best_path_nodes = None
        self.feature_names = None
        
    def _evaluate_node(self, X, y, node_id, path, current_path, current_depth=0):
        if current_depth >= self.max_depth:
            return
            
        mask = path(X)
        node_samples = y[mask]
        
        if len(node_samples) >= self.min_cluster_size:
            purity = np.mean(node_samples)
            score = purity * (len(node_samples) / self.min_cluster_size)
            
            if purity >= self.min_purity and score > self.best_score:
                self.best_score = score
                self.best_path = path
                self.best_path_nodes = current_path.copy()
    
    def fit(self, X, y, feature_names=None):
        """
        Fit the model to find high-density clusters
        
        Parameters:
        feature_names: list of strings, names of the features (optional)
        """
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        self.tree.fit(X, y)
        
        def make_path(node_indices):
            def path_func(X):
                return self.tree.decision_path(X).toarray()[:, node_indices].all(axis=1)
            return path_func
        
        n_nodes = self.tree.tree_.node_count
        children_left = self.tree.tree_.children_left
        children_right = self.tree.tree_.children_right
        
        def traverse_tree(node_id=0, current_path=None, depth=0):
            if current_path is None:
                current_path = []
                
            if depth >= self.max_depth:
                return
                
            current_path.append(node_id)
            path_func = make_path(current_path)
            
            self._evaluate_node(X, y, node_id, path_func, current_path, depth)
            
            if children_left[node_id] != -1:
                traverse_tree(children_left[node_id], current_path.copy(), depth + 1)
                traverse_tree(children_right[node_id], current_path.copy(), depth + 1)
                
        traverse_tree()
        return self
        
    def predict(self, X):
        if self.best_path is None:
            return np.zeros(len(X))
        return self.best_path(X).astype(int)
    
    def get_cluster_stats(self, X, y):
        if self.best_path is None:
            return None
            
        mask = self.best_path(X)
        cluster_samples = y[mask]
        
        return {
            'cluster_size': len(cluster_samples),
            'purity': np.mean(cluster_samples),
            'total_positives': np.sum(cluster_samples),
            'score': self.best_score
        }
    
    def get_decision_rules(self):
        """
        Returns the decision rules that define the border of the identified cluster
        """
        if self.best_path_nodes is None:
            return "No cluster found"
            
        tree = self.tree.tree_
        feature = tree.feature
        threshold = tree.threshold
        
        rules = []
        for node_id in self.best_path_nodes:
            # Skip leaf nodes
            if feature[node_id] == -2:
                continue
                
            # Get the feature name and threshold for this node
            feature_name = self.feature_names[feature[node_id]]
            node_threshold = threshold[node_id]
            
            # Determine if this is a left or right split
            is_left = False
            if node_id > 0:
                parent = (node_id - 1) // 2
                if tree.children_left[parent] == node_id:
                    is_left = True
            
            # Create the rule string
            if is_left:
                rule = f"{feature_name} <= {node_threshold:.3f}"
            else:
                rule = f"{feature_name} > {node_threshold:.3f}"
            
            rules.append(rule)
        
        return " AND ".join(rules)

# Create and fit the classifier
clf = HighDensityClusterClassifier(
    min_cluster_size=400,  # Minimum cluster size you want
    min_purity=0.53,       # Minimum percentage of 1s in the cluster
    max_depth=3           # Maximum depth of the tree
)


df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data_9_in_10W.csv')

# Prepare features (X) and target (y)
exclude_columns = ['target', 'stock', 'date']
feature_columns = [col for col in df.columns if col not in exclude_columns]

X = df[feature_columns]
y = df['target']

# Fit the model
clf.fit(X, y,feature_names=feature_columns)

rules = clf.get_decision_rules()
print("\nDecision Rules for the High-Density Cluster:")
print(rules)

# Get predictions (1 for samples in the high-density cluster, 0 otherwise)
predictions = clf.predict(X)

# Get statistics about the found cluster
stats = clf.get_cluster_stats(X, y)
print(stats)