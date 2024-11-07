import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .utils import *


"""
to run : py -m experiment1.sandbox
"""
def find_best_fit(vectorY):
    """
    Find whether the data fits better to a linear (y=ax+b) or quadratic (y=ax^2+bx+c) equation
    """
    # Create x vector
    vectorX = np.array(range(len(vectorY))).reshape(-1, 1)
    vectorY = np.array(vectorY)
    
    # Fit linear regression (y = ax + b)
    print("linear regression : ")
    print("vectorX : ",vectorX)
    print("vectorY : ",vectorY)
    linear_reg = LinearRegression()
    linear_reg.fit(vectorX, vectorY)
    linear_pred = linear_reg.predict(vectorX)
    linear_score = r2_score(vectorY, linear_pred)
    print("linear_reg.coef_ : ",linear_reg.coef_)
    print("intercept_ : ",linear_reg.intercept_)
    print("linear_pred : ",linear_pred)
    print("linear_score : ",linear_score)


    # Fit quadratic regression (y = ax² + bx + c)
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(vectorX)
    print("X_poly")
    print(X_poly)
    quad_reg = LinearRegression()
    quad_reg.fit(X_poly, vectorY)
    quad_pred = quad_reg.predict(X_poly)
    quad_score = r2_score(vectorY, quad_pred)
    print("Quadratic_reg.coef_ : ",quad_reg.coef_)
    print("intercept_ : ",quad_reg.intercept_)
    print("quad_pred : ",quad_pred)
    print("quad_score : ",quad_score)


    if max(quad_score,linear_score)<=0.7:
        return max(quad_score,linear_score),""
    # Compare scores and return result
    if linear_score > quad_score:
        return linear_score,{
            'best_fit': 'linear',
            'score': linear_score,
            'equation': f'y = {linear_reg.coef_[0]}x + {linear_reg.intercept_}'
        }
    else:
        return quad_score,{
            'best_fit': 'quadratic',
            'score': quad_score,
            'equation': f'y = {quad_reg.coef_[2]}x² + {quad_reg.coef_[1]}x + {quad_reg.intercept_}'
        }
    

print("hey")
vectorY =[0.4960352422907489, 0.48546255506607927, 0.4691358024691358, 0.46255506607929514, 0.44757709251101324, 0.4532627865961199, 0.4643171806167401, 0.4687224669603524]
score,result = find_best_fit(vectorY)
print(result)