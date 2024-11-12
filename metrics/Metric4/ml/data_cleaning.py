import pandas as pd
import random
#from ..utils import *

"""
to run : py -m metrics.Metric4.ml.data_cleaning
"""
training_data_path = 'C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric4/ml/training_data_raw.csv'
# Read the CSV file
df = pd.read_csv(training_data_path)



# Remove duplicates
df.drop_duplicates(inplace=True)
df = df.dropna()



# Save the processed data to a new CSV file
df.to_csv('cleaned_training_data.csv', index=False)

print(f"Original shape: {pd.read_csv(training_data_path).shape}")
print(f"Processed shape: {df.shape}")
print(f"Number of rows with to_buy = 0: {(df['to_buy'] == 0).sum()}")
print(f"Number of rows with to_buy = 1: {(df['to_buy'] == 1).sum()}")