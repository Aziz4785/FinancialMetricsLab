import pandas as pd
import random
#from ..utils import *

"""
to run : py -m experiment1.cleaning
"""
training_data_path = 'C:/Users/aziz8/Documents/FinancialMetricsLab/experiment1/raw_data_10_in_2M.csv'
# Read the CSV file
df = pd.read_csv(training_data_path)


#df=df.drop(['PriceToRD'], axis=1)
print({
    'GPRatio_nulls': df['GPRatio'].isna().sum(),
    'R&D_nulls': df['i_researchAndDevelopmentExpenses'].isna().sum(),
    'FCF_nulls': df['c_freeCashFlow'].isna().sum(),
    'i_revenue': df['i_revenue'].isna().sum(),
})

print({
    'GPRatio_empty': (df['GPRatio'] == '').sum(),
    'R&D_empty': (df['i_researchAndDevelopmentExpenses'] == '').sum(),
    'FCF_empty': (df['c_freeCashFlow'] == '').sum(),
    'i_revenue_empty': (df['i_revenue'] == '').sum()
})

# Remove duplicates
df.drop_duplicates(inplace=True)
df = df.dropna()

stocks = df['stock'].unique()
print(f"number of unique stocks : {len(stocks)}")

count_0 = (df['target'] == 0).sum()
count_1 = (df['target'] == 1).sum()
print(f" count 0 : {count_0}")
print(f" count 1 : {count_1}")
rows_to_remove = count_1 - count_0
if rows_to_remove > 0:
    indices_to_remove = df[df['target'] == 1].index
    indices_to_remove = random.sample(list(indices_to_remove), rows_to_remove)
    df = df.drop(indices_to_remove)

# Save the processed data to a new CSV file
df.to_csv('cleaned_data_10_in_2M.csv', index=False)

print(f"Original shape: {pd.read_csv(training_data_path).shape}")
print(f"Processed shape: {df.shape}")
print(f"Number of rows with to_buy = 0: {(df['target'] == 0).sum()}")
print(f"Number of rows with to_buy = 1: {(df['target'] == 1).sum()}")