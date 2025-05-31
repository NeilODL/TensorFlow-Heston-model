import pandas as pd
import os

# Define the file paths of the four CSV files
file_paths = [
    '/Users/neildrew-lopez/Desktop/Project/Data sets/FDM250kv1.csv',
    '/Users/neildrew-lopez/Desktop/Project/Data sets/FDM250kv2.csv',
    '/Users/neildrew-lopez/Desktop/Project/Data sets/FDM250kv3.csv',
    '/Users/neildrew-lopez/Desktop/Project/Data sets/FDM250kv4.csv'
]

# Create an empty list to store DataFrames
df_list = []

# Loop through each file path and read the CSV file into a DataFrame
for file_path in file_paths:
    df = pd.read_csv(file_path)  # Read the CSV file
    df_list.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# Define the output file name
output_file = 'parameter_option_prices.csv'

# Save the combined DataFrame to the current working directory
combined_df.to_csv(output_file, index=False)

print(f"Combined data saved as {output_file} in the current working directory.")
