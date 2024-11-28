import pandas as pd

"""
to run : py -m experiment1.sandbox2  (if it doesnt work add.py)
"""

df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/cleaned_data_15_in_3M.csv')
exclude_columns = ['target', 'stock', 'date']
feature_columns = [col for col in df.columns if col not in exclude_columns]

df = df[feature_columns]

correlations = df.corr()

# Extract correlations with 'market_cap'
market_cap_correlations = correlations['i_netIncome'].sort_values(ascending=False)

# Display the top correlated columns
print(market_cap_correlations)