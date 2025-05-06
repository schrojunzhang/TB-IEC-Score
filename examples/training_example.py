#!/usr/bin/env python3
"""
Example script for training a TB-IEC-Score model
"""
import argparse
import os

from tb_iecs.core.pipeline import TBIECPipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a TB-IEC-Score model")
    
    parser.add_argument("--protein_file", type=str, required=True,
                        help="Path to protein PDB file")
    parser.add_argument("--crystal_ligand_file", type=str, required=True,
                        help="Path to crystal ligand MOL2 file")
    parser.add_argument("--ligand_path", type=str, required=True,
                        help="Path to directory containing ligand SDF files")
    parser.add_argument("--label_csv", type=str, required=True,
                        help="Path to CSV file with ligand labels (name, label)")
    parser.add_argument("--dst_dir", type=str, default="./results",
                        help="Directory for results (default: ./results)")
    parser.add_argument("--model_file", type=str, default="./tb_iecs_model.pkl",
                        help="Path to save model (default: ./tb_iecs_model.pkl)")
    parser.add_argument("--model_type", type=str, default="xgboost",
                        choices=["xgboost", "svm", "rf"],
                        help="Model type to use (default: xgboost)")
    parser.add_argument("--no_hyper_opt", action="store_true",
                        help="Disable hyperparameter optimization")
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.dst_dir, exist_ok=True)
    
    print("Starting TB-IEC-Score model training:")
    print(f"Protein file: {args.protein_file}")
    print(f"Crystal ligand file: {args.crystal_ligand_file}")
    print(f"Ligand directory: {args.ligand_path}")
    print(f"Label CSV: {args.label_csv}")
    print(f"Model type: {args.model_type}")
    print(f"Hyperparameter optimization: {not args.no_hyper_opt}")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = TBIECPipeline(
        protein_file=args.protein_file,
        crystal_ligand_file=args.crystal_ligand_file,
        dst_dir=args.dst_dir
    )
    
    # Train model
    print("Training model...")
    metrics = pipeline.train(
        ligand_path=args.ligand_path,
        label_csv=args.label_csv,
        model_file=args.model_file,
        model_type=args.model_type,
        hyper_opt=not args.no_hyper_opt
    )
    
    # Print evaluation metrics
    print("\nTraining complete. Evaluation metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key:20}: {value:.4f}")
    
    print(f"\nModel saved to: {args.model_file}")


if __name__ == "__main__":
    main() 