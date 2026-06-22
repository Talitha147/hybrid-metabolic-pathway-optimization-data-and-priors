import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def add_noise_to_dataset(input_dir, output_dir, noise_percent, seed=42):

    np.random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")

    # Scale factor for noise
    noise_scale = noise_percent / 100.0
    
    csv_files = list(input_path.glob("*.csv"))
    
    for csv_file in csv_files:
        if "strain_designs" in csv_file.name:
            print(f"Copying {csv_file.name} without noise...")
            df = pd.read_csv(csv_file)
            df.to_csv(output_path / csv_file.name, index=False)
            continue
            
        print(f"Adding {noise_percent}% noise to {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Identify data columns (all columns except 'time')
        data_cols = [col for col in df.columns if col != 'time']
        
        if not data_cols:
            print(f"No data columns found in {csv_file.name}, skipping.")
            df.to_csv(output_path / csv_file.name, index=False)
            continue
            
        data = df[data_cols].values
        
      
        std_dev = np.std(data)
        if std_dev == 0:
            std_dev = 1.0
            
        sigma = noise_scale * std_dev
        print(f"  - Signal StdDev: {std_dev:.4e}, Noise sigma: {sigma:.4e}")
        
        noise = np.random.normal(0, sigma, size=data.shape)
        noisy_data = data + noise
        
        # Ensure no negative concentrations
        noisy_data = np.maximum(noisy_data, 0.0)
        
        # Update dataframe
        df[data_cols] = noisy_data
        
        # Save to new location
        df.to_csv(output_path / csv_file.name, index=False)
        print(f"Saved noisy dataset to {output_path / csv_file.name}")

if __name__ == "__main__":
    
    add_noise_to_dataset("data", "data_noise_20", 20, 42)
