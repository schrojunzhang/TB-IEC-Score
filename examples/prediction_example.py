#!/usr/bin/env python3
"""
Example script for using a trained TB-IEC-Score model for prediction
"""
import argparse
import os

import pandas as pd
from tb_iecs.core.pipeline import TBIECPipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Use a trained TB-IEC-Score model for prediction")
    
    parser.add_argument("--protein_file", type=str, required=True,
                        help="Path to protein PDB file")
    parser.add_argument("--crystal_ligand_file", type=str, required=True,
                        help="Path to crystal ligand MOL2 file")
    parser.add_argument("--ligand_path", type=str, required=True,
                        help="Path to directory containing ligand SDF files")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to trained model file")
    parser.add_argument("--dst_dir", type=str, default="./results",
                        help="Directory for results (default: ./results)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save prediction results (default: dst_dir/result.csv)")
    parser.add_argument("--print_top", type=int, default=10,
                        help="Number of top compounds to print (default: 10)")
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.dst_dir, exist_ok=True)
    
    # Set output CSV path
    output_csv = args.output_csv or os.path.join(args.dst_dir, "result.csv")
    
    print("Starting TB-IEC-Score prediction:")
    print(f"Protein file: {args.protein_file}")
    print(f"Crystal ligand file: {args.crystal_ligand_file}")
    print(f"Ligand directory: {args.ligand_path}")
    print(f"Model file: {args.model_file}")
    print(f"Output CSV: {output_csv}")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = TBIECPipeline(
        protein_file=args.protein_file,
        crystal_ligand_file=args.crystal_ligand_file,
        dst_dir=args.dst_dir
    )
    
    # Run prediction
    print("Running prediction...")
    results = pipeline.predict(
        ligand_path=args.ligand_path,
        model_file=args.model_file
    )
    
    # Print summary
    print("\nPrediction complete.")
    print("-" * 50)
    total_compounds = len(results)
    active_compounds = len(results[results['pred_y'] == 1])
    print(f"Total compounds: {total_compounds}")
    print(f"Predicted active: {active_compounds} ({active_compounds/total_compounds*100:.2f}%)")
    print(f"Predicted inactive: {total_compounds - active_compounds} ({(total_compounds - active_compounds)/total_compounds*100:.2f}%)")
    
    # Print top compounds
    print(f"\nTop {args.print_top} compounds by score:")
    print("-" * 50)
    top_compounds = results.sort_values(by='pred_y_proba', ascending=False).head(args.print_top)
    for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
        status = "ACTIVE" if row['pred_y'] == 1 else "INACTIVE"
        print(f"{i:2d}. {row['name']:<20} - Score: {row['pred_y_proba']:.4f} - Prediction: {status}")
    
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main() 