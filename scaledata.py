# scale_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def scale_features(input_file='filtered_option_prices.csv',
                   output_file='scaled_option_prices.csv',
                   scaler_save_path='scaler.save',
                   feature_columns=['moneyness', 'initial_variance', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma']):
    """
    Reads the input CSV file, scales the specified feature columns using StandardScaler,
    and saves the scaled data to the output CSV file.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the scaled CSV file.
    - scaler_save_path (str): Path to save the fitted scaler object.
    - feature_columns (list): List of feature column names to scale.
    """
    # Load the data
    df = pd.read_csv(input_file)
    print(f"Loaded data from '{input_file}' with shape {df.shape}")

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the feature columns and transform
    df_features = df[feature_columns]
    df_scaled_features = scaler.fit_transform(df_features)

    # Save the scaler for future use
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to '{scaler_save_path}'")

    # Create a new DataFrame with scaled features
    df_scaled = df.copy()
    df_scaled[feature_columns] = df_scaled_features

    # Save the scaled data to the output file
    df_scaled.to_csv(output_file, index=False)
    print(f"Scaled data saved to '{output_file}'")

if __name__ == "__main__":
    scale_features()
