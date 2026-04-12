from __future__ import annotations

import pandas as pd
from src.utils import load_config

def main() -> None:
    """Main pipeline for data preprocessing."""
    config = load_config()
    print(f"Preprocessing data for {config['project']['name']}...")
    
    # Logic to load raw data, clean, and save processed artifacts
    # preprocessor = TweetPreprocessor(
    #     model_name=config["model"]["name"],
    #     max_length=config["data"]["max_length"]
    # )
    
    print("Pre-processing complete.")

if __name__ == "__main__":
    main()
