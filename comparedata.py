import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for better aesthetics
sns.set(style="whitegrid")

# File paths (assuming they are in the current directory)
fourier_file = 'Fourier_Prices.csv'
fsm_file = 'filtered_option_prices.csv'

# Read the datasets
# For Fourier_Prices.csv
fourier_df = pd.read_csv(fourier_file)

# For parameter_option_prices.csv, reorder columns to match Fourier dataset
fsm_df = pd.read_csv(fsm_file)
fsm_df = fsm_df[['moneyness', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma', 'initial_variance', 'option_price']]

# Function to compute and print mean and standard deviation
def print_stats(df, dataset_name):
    print(f"\nStatistics for {dataset_name}:")
    stats = df.describe().loc[['mean', 'std']].transpose()
    print(stats[['mean', 'std']])

# 1. Print mean and standard deviation for each parameter
print_stats(fourier_df, 'Fourier_Prices.csv')
print_stats(fsm_df, 'parameter_option_prices.csv')

