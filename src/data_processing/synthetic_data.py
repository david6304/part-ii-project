import bnlearn as bn
import pandas as pd
from src.utils.config_loader import load_config

def generate_synthetic_dataset(dataset_name, save: bool = True, n=10000, variant="base") -> pd.DataFrame:
    """Generate and save synthetic dataset"""
    config = load_config()
    df = bn.import_example(dataset_name).astype(float)
    
    if save:
        output_dir = config['data']['synthetic'][dataset_name]
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{dataset_name}_{variant}.csv", index=False)
    
    return df

if __name__ == "__main__":
    datasets = ["asia", "alarm", "auto_mpg"]
    
    for dataset in datasets:
        generate_synthetic_dataset(dataset_name=dataset)
        print(f"Generated synthetic dataset: {dataset}, Variant: base, n=10000")