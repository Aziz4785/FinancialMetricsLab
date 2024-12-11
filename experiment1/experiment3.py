import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .utils import *
from scipy import stats
"""
to run : py -m experiment1.experiment3
"""
df = pd.read_csv('experiment1/raw_data_36D_in_6W.csv')
df = df.dropna()
#df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data_augmented_0.csv')

#df = df[df['dividend_payout_ratio'] >=0]
#df = df[df['i_grossProfit'] <= 293564000.00]
df = df[df['6M_return'] <= -7.86]
df = df[df['GPRatio'] >=0.23]
df = df[df['price'] <= 45.86]
#df = df[df['price_to_sales'] <= 7.74] #KEEP
#df = df[df['cashflow_to_debt']>=0.14] #KEEP
print("length of df : ",len(df))
percentage_target_1 = (df['target'].value_counts(normalize=True).get(1, 0)) * 100
print(f"Percentage of rows where target == 1: {percentage_target_1:.2f}%")

def find_best_fit(vectorY):
    """
    Find whether the data fits better to a linear (y=ax+b) or quadratic (y=ax^2+bx+c) equation
    """
    # Create x vector
    vectorX = np.array(range(len(vectorY))).reshape(-1, 1)
    vectorY = np.array(vectorY)
    
    # Fit linear regression (y = ax + b)
    linear_reg = LinearRegression()
    linear_reg.fit(vectorX, vectorY)
    linear_pred = linear_reg.predict(vectorX)
    linear_score = r2_score(vectorY, linear_pred)
    n = len(vectorY)


    # Fit quadratic regression (y = ax² + bx + c)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(vectorX)
    quad_reg = LinearRegression()
    quad_reg.fit(X_poly, vectorY)
    quad_pred = quad_reg.predict(X_poly)
    quad_score = r2_score(vectorY, quad_pred)
    
    if max(quad_score,linear_score)<=0.7:
        return max(quad_score,linear_score),"",0
    # Compare scores and return result
    if linear_score > quad_score:
        return linear_score,{
            'best_fit': 'linear',
            'score': linear_score,
            'equation': f'y = {linear_reg.coef_[0]}x + {linear_reg.intercept_}'
        },linear_reg.coef_[0]
    else:
        return quad_score,{
            'best_fit': 'quadratic',
            'score': quad_score,
            'equation': f'y = {quad_reg.coef_[2]}x² + {quad_reg.coef_[1]}x + {quad_reg.intercept_}'
        },max(quad_reg.coef_[2],quad_reg.coef_[1])
    
def print_percentiles(df):
    # Skip specified columns
    columns_to_skip = ['to_buy', 'date', 'stock', 'symbol', 'target']
    
    for column in df.columns:
        if column not in columns_to_skip:
            p15 = df[column].quantile(0.05)
            p85 = df[column].quantile(0.95)
            #print(f"{column}:")
            #print(f"  10th percentile: {p15:.2f}")
            #print(f"  90th percentile: {p85:.2f}")

            # Filter values between p15 and p85
            filtered_values = df[column][(df[column] >= p15) & (df[column] <= p85)]
            if len(filtered_values)<100:
                print("too few values to perform analysis")
                continue
            # Calculate segments using percentiles
            #segments = [0, 33.33,66.66, 100]
            segments = [0, 25, 50, 75, 100]
            #segments = [0, 20, 40, 60, 80, 100 ]
            #segments = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
            #segments = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            segment_values = [filtered_values.quantile(p/100) for p in segments]

            # Create list of valid segments (where boundaries are different)
            valid_segments = []
            valid_segment_indices = []
            for i in range(len(segments)-1):
                if segment_values[i] != segment_values[i+1]:
                    valid_segments.append((segment_values[i], segment_values[i+1]))
                    valid_segment_indices.append(i)
            
            if valid_segments:
                #print("\n  Segments boundaries:")
                #for idx, (left, right) in enumerate(valid_segments, 1):
                    #print(f"    Segment {idx}: {left:.2f} to {right:.2f}")
                
                vectorY = []
                #print("\n  Segments statistics:")
                max_ratio=0
                min_ratio =100
                for idx, (left, right) in enumerate(valid_segments, 1):
                    # Get mask for values in this segment
                    segment_mask = (df[column] >= left) & (df[column] <= right)
                    
                    # Count total elements in segment
                    total_in_segment = segment_mask.sum()
                    
                    # Count elements with target=1 in segment
                    target_in_segment = ((df['target'] == 1) & segment_mask).sum()
                    
                    # Calculate ratio (handle division by zero)
                    ratio = (target_in_segment / total_in_segment) if total_in_segment > 0 else 0
                    max_ratio = max(max_ratio,ratio)
                    min_ratio= min(min_ratio,ratio)
                    vectorY.append(ratio)
                    # print(f"    Segment {idx}:")
                    # print(f"      Total elements: {total_in_segment}")
                    # print(f"      Elements with target=1: {target_in_segment}")
                    # print(f"      Target ratio: {ratio:.2%}")

                score,result,max_coeff = find_best_fit(vectorY)

                
    

                if score >0.9 and max_ratio>0.6 and (max_ratio-min_ratio)>0.1:
                    print()
                    print(f"{column}:")
                    print(f"we divide the values of {column} into equal sized segments:")
                    for idx, (left, right) in enumerate(valid_segments, 1):
                        print(f"    Segment {idx}: {left:.2f} to {right:.2f}")
                    print("then for each segment we calculate the proportion of target==1 : ")
                    print(vectorY)
                    print("then we see if this follow a pattern")
                    print(result)
                    

            else:
                print("\n  No valid segments found (all boundaries are equal)")
print_percentiles(df)