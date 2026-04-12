from __future__ import annotations

import torch
from src.utils import load_config, get_device
from src.model.loader import load_checkpoint

def main() -> None:
    """Main pipeline for model evaluation."""
    config = load_config()
    device = get_device()
    
    print(f"Evaluating model for {config['project']['name']} on {device}...")
    
    
    # Load best checkpoint
    # model = load_checkpoint(model, "path/to/best.pt", device)
    
    print("Evaluation pipeline setup complete.")

if __name__ == "__main__":
    main()
