import pandas as pd

# Load the dataset
input_file = 'parameter_option_prices.csv'
df = pd.read_csv(input_file)

# Filter rows where option_price is above 10
filtered_df = df[df['option_price'] < 0.66]

# Save the filtered data to a new file, overwriting if it exists
output_file = 'filtered_option_prices.csv'
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")