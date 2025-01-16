import os
import numpy as np
import pandas as pd
import bnlearn as bn
from pgmpy.utils import get_example_model

# Configurations
OUTPUT_DIR = "data/synthetic"
SAMPLE_SIZES = [1000, 5000, 10000]
FLIP_PERCENTAGES = [0.05, 0.1, 0.2]  # Random flips for categorical data
GAUSSIAN_NOISE_STDS = [0.05, 0.1, 0.2]  # Standard deviations for continuous noise
MISSING_RATES = [0.1, 0.3, 0.5]  # Percentage of missing features

# Example datasets
data = {
    "asia": [],
    "alarm": [],
    "auto_mpg": [],
}
for size in SAMPLE_SIZES:
    data["asia"].append(bn.import_example("asia", n=size))
    data["alarm"].append(bn.import_example("alarm", n=size))
    data["auto_mpg"].append(bn.import_example("auto_mpg", n=size))


# Utility functions
def save_data(data, path, filename):
    """Save data to CSV."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    data.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")

def add_random_flips(data, flip_prob):
    """Randomly flip categorical variables."""
    flipped_data = data.copy()
    for col in flipped_data.columns:
        if flipped_data[col].dtype.name == 'category':  # Only flip categorical columns
            unique_values = flipped_data[col].unique()
            flip_mask = np.random.rand(len(flipped_data)) < flip_prob
            flipped_data.loc[flip_mask, col] = np.random.choice(unique_values, size=flip_mask.sum())
    return flipped_data

def add_gaussian_noise(data, std_dev):
    """Add Gaussian noise to continuous variables."""
    noisy_data = data.copy()
    print(noisy_data[:10])
    for col in noisy_data.columns:
        if noisy_data[col].dtype.name != 'category':  # Only add noise to continuous columns
            noisy_data[col] += np.random.normal(0, std_dev, size=len(noisy_data))
    return noisy_data

def mask_data(data, missing_rate):
    """Mask random features to simulate missing data."""
    masked_data = data.copy()
    mask = np.random.rand(*masked_data.shape) < missing_rate
    return masked_data.mask(mask)


def do_intervention(data, node, value):
    """Perform do-interventions by fixing a node's value."""
    intervened_data = data.copy()
    intervened_data[node] = value
    return intervened_data

# Main Function to Generate Data
def generate_synthetic_data():
    for name, datasets in data.items():
        for dataset, size in zip(datasets, SAMPLE_SIZES):
            # Save original data
            save_data(dataset, OUTPUT_DIR + f"/{name}/base", f"{name}_{size}_original.csv")

            # Add random flips
            if name != "auto_mpg":
                for flip_prob in FLIP_PERCENTAGES:
                    flipped_data = add_random_flips(dataset, flip_prob)
                    save_data(flipped_data, OUTPUT_DIR + f"/{name}/flipped", f"{name}_{size}_flipped_{flip_prob}.csv")
            # Add Gaussian noise
            else: # Only add noise to continuous data
                for std_dev in GAUSSIAN_NOISE_STDS:
                    noisy_data = add_gaussian_noise(dataset, std_dev)
                    save_data(noisy_data, OUTPUT_DIR + f"/{name}/gaussian", f"{name}_{size}_gaussian_{std_dev}.csv")

            # Mask data
            for missing_rate in MISSING_RATES:
                masked_data = mask_data(dataset, missing_rate)
                save_data(masked_data, OUTPUT_DIR + f"/{name}/missing", f"{name}_{size}_missing_{missing_rate}.csv")
                
# Run the data generation
if __name__ == "__main__":
    generate_synthetic_data()