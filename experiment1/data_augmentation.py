"""
augmentation : 2 type of operations :
addition division

feature lvl 0 are the primary feature of the csv

so we will do

lvl 0 -> augmentation -> lvl 1 - ---->augmentation --> lvl 2
                                |
      --------------------------|-->augmentation -> lvl 1_0
"""

"""
to run : py -m experiment1.data_augmentation
"""

import pandas as pd
import numpy as np
from itertools import combinations
import csv
import json
import random

AUGMENTATION_LVL =0 #0 then 1
 #to balance the datast before saving it (only put true at the last augmentation level)
FEATURES_TO_ADD = 1300 

BALANCE_DATASET = True

def get_mult_magnitude(df, feature1, feature2):
    avg1 = abs(df[feature1]).mean()
    avg2 = abs(df[feature2]).mean()
    ratio = max(avg1/avg2 if avg2 != 0 else float('inf'), 
                avg2/avg1 if avg1 != 0 else float('inf'))
    return ratio

def get_abs_max(series):
        return abs((series).max())

def generate_new_features(df, feature_columns,n_new_features):
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Get all numeric columns
    numeric_cols = feature_columns
    
    # Function to check if a feature has more than 20% zeros
    def has_many_zeros(series):
        return (series == 0).mean() > 0.1
    
    # Function to check if a feature has more than 20% negative values
    def has_many_negatives(series):
        return (series < 0).mean() > 0.1
    
    def has_too_many_nans(series):
        return series.isna().mean() > 0.02
    
    # Function to perform random operation
    def random_operation(series1, series2, op):
        
        if op == 'add':
            return series1 + series2,'add'
        elif op == 'sub':
            return series1 - series2,'sub'
        else:  # divide
            # First check for denominator being 0.0
            result = pd.Series(np.nan, index=series1.index)  # initialize with NaN
            # Where denominator is not 0.0, perform division
            non_zero_denom = series2 != 0.0
            # For non-zero denominators:
            # - If numerator is 0.0, result is 0
            # - Otherwise perform division
            result[non_zero_denom] = np.where(
                series1[non_zero_denom] == 0.0,
                0.0,
                series1[non_zero_denom] / series2[non_zero_denom]
            )
            return result, 'div'
        
    # Counter for new feature names
    feature_count = 1
    
    new_features_expl = {}
    used_pairs = set()
    # Generate n_new_features new columns
    while feature_count <= n_new_features:
        feature1, feature2 = np.random.choice(numeric_cols, size=2, replace=False)
        
        # Check conditions and perform operations
        if has_many_zeros(df[feature1]) or has_many_zeros(df[feature2]):
            # If one feature has many zeros, perform addition
            operations = ['add', 'sub']
            op = np.random.choice(operations)
            uniqueoperation = f'{feature1}_{feature2}_{op}'
            if uniqueoperation in used_pairs:
                feature_count+=1
                continue
            else:
                used_pairs.add(uniqueoperation)

            new_feature,ope = random_operation(df[feature1], df[feature2], op)

        elif get_mult_magnitude(df, feature1, feature2)>100000000000:
            continue
        elif get_mult_magnitude(df, feature1, feature2)>1000000 :
            avg1 = abs(df[feature1]).mean()
            avg2 = abs(df[feature2]).mean()
            ope = 'div'

            if avg1 > avg2:
                new_feature = df[feature1] / df[feature2]
                uniqueoperation = f'{feature1}_{feature2}_{ope}'
            else:
                new_feature = df[feature2] / df[feature1]
                uniqueoperation = f'{feature2}_{feature1}_{ope}'

            if uniqueoperation in used_pairs:
                feature_count+=1
                continue
            else:
                used_pairs.add(uniqueoperation)
               
        elif has_many_negatives(df[feature1]) or has_many_negatives(df[feature2]):
            # If one feature has many negatives, perform random addition or division
            operations = ['add', 'div']
            op = np.random.choice(operations)
            new_feature,ope = random_operation(df[feature1], df[feature2], op)
            uniqueoperation = f'{feature1}_{feature2}_{op}'
            if uniqueoperation in used_pairs:
                feature_count+=1
                continue
            else:
                used_pairs.add(uniqueoperation)
        else:
            # Otherwise perform random operation (add, subtract, or divide)
            operations = ['add', 'sub', 'div']
            op = np.random.choice(operations)
            new_feature,ope = random_operation(df[feature1], df[feature2], op)
            uniqueoperation = f'{feature1}_{feature2}_{op}'
            if uniqueoperation in used_pairs:
                feature_count+=1
                continue
            else:
                used_pairs.add(uniqueoperation)
        
        if has_too_many_nans(new_feature):
            print(f"{feature1}_{feature2}_{ope} poduces too many nans")
            continue
        if ope =='div' and get_abs_max(new_feature) >1000000:
            continue
        # Create new column name
        new_col_name = f'{feature_count}_feature_{AUGMENTATION_LVL}'
        new_features_expl[feature_count] =f'{feature1}_{feature2}_{ope}'
        # Add new feature to dataframe
        result_df[new_col_name] = new_feature
        
        feature_count += 1
    
    return result_df,new_features_expl

if AUGMENTATION_LVL ==0:
    df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data.csv')
else:
    df = pd.read_csv(f'C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data_augmented_{AUGMENTATION_LVL-1}.csv')
exclude_columns = ['target', 'stock', 'date']
feature_columns = [col for col in df.columns if col not in exclude_columns]
new_df,new_features_expl = generate_new_features(df,feature_columns, n_new_features=FEATURES_TO_ADD)
new_df = new_df.dropna()
if BALANCE_DATASET:
    count_0 = (new_df['target'] == 0).sum()
    count_1 = (new_df['target'] == 1).sum()
    print(f" count 0 : {count_0}")
    print(f" count 1 : {count_1}")
    rows_to_remove = count_1 - count_0
    if rows_to_remove > 0:
        indices_to_remove = new_df[new_df['target'] == 1].index
        indices_to_remove = random.sample(list(indices_to_remove), rows_to_remove)
        new_df = new_df.drop(indices_to_remove)

file_name = f'cleaned_data_augmented_{AUGMENTATION_LVL}.csv'
new_df.to_csv(file_name, index=False)

with open(f'new_features_expl_{AUGMENTATION_LVL}.json', 'w') as json_file:
    json.dump(new_features_expl, json_file, indent=4) 