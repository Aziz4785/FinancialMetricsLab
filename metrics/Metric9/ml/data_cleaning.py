import pandas as pd
import random
#from ..utils import *

"""
to run : py -m metrics.Metric9.ml.data_cleaning
"""

training_data_path = 'metrics/Metric9/ml/training_data_raw.csv'
# Read the CSV file
df = pd.read_csv(training_data_path)

print("length of df : ",len(df))
print(df.head())
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] > '2019-11-10']

# Remove duplicates
df.drop_duplicates(inplace=True)
df = df.dropna()



# Save the processed data to a new CSV file
df.to_csv('metrics/Metric9/ml/cleaned_training_data.csv', index=False)

print(f"Original shape: {pd.read_csv(training_data_path).shape}")
print(f"Processed shape: {df.shape}")
print(f"Number of rows with to_buy = 0: {(df['to_buy'] == 0).sum()}")
print(f"Number of rows with to_buy = 1: {(df['to_buy'] == 1).sum()}")