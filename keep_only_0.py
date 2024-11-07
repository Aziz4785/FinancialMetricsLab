import pandas as pd

# Read the CSV file
df = pd.read_csv('training_data.csv')

# Filter rows where to_buy equals 0
filtered_df = df[df['to_buy'] == 0]

# Save the filtered data to a new CSV file
filtered_df.to_csv('only0_training_data.csv', index=False)