#!/usr/bin/env python3
"""
Command-line interface for TB-IEC-Score
"""
import argparse
import os
import sys
import logging
from tb_iecs.core.pipeline import TBIECPipeline


def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def train_command(args):
    """
    Handle the train subcommand.
    
    Args:
        args: Command line arguments
    """
    # Initialize the pipeline
    pipeline = TBIECPipeline(
        protein_file=args.protein_file,
        crystal_ligand_file=args.crystal_ligand_file,
        dst_dir=args.dst_dir,
        smina_path=args.smina_path,
        nnscore_path=args.nnscore_path,
        mgltool_path=args.mgltool_path
    )
    
    # Train the model
    metrics = pipeline.train(
        ligand_path=args.ligand_path,
        label_csv=args.label_csv,
        model_file=args.model_file,
        model_type=args.model_type,
        hyper_opt=not args.no_hyper_opt
    )
    
    # Print metrics
    print("\nTraining Results:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:15}: {value:.4f}")
    
    print(f"\nModel saved to: {args.model_file}")


def predict_command(args):
    """
    Handle the predict subcommand.
    
    Args:
        args: Command line arguments
    """
    # Initialize the pipeline
    pipeline = TBIECPipeline(
        protein_file=args.protein_file,
        crystal_ligand_file=args.crystal_ligand_file,
        dst_dir=args.dst_dir,
        smina_path=args.smina_path,
        nnscore_path=args.nnscore_path,
        mgltool_path=args.mgltool_path
    )
    
    # Make predictions
    results = pipeline.predict(
        ligand_path=args.ligand_path,
        model_file=args.model_file
    )
    
    # Print summary
    active_count = len(results[results['pred_y'] == 1])
    inactive_count = len(results[results['pred_y'] == 0])
    total_count = len(results)
    
    print("\nPrediction Results:")
    print("-" * 40)
    print(f"Total ligands:      {total_count}")
    print(f"Predicted active:   {active_count} ({active_count/total_count*100:.1f}%)")
    print(f"Predicted inactive: {inactive_count} ({inactive_count/total_count*100:.1f}%)")
    print(f"\nResults saved to: {os.path.join(args.dst_dir, 'result.csv')}")


def main():
    """Main entry point for the CLI"""
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="TB-IEC-Score: An accurate and efficient machine learning-based scoring function for virtual screening"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        title="Commands", dest="command", help="Command to execute"
    )
    subparsers.required = True
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--protein_file", type=str, required=True,
        help="Path to protein PDB file"
    )
    common_parser.add_argument(
        "--crystal_ligand_file", type=str, required=True,
        help="Path to crystal ligand MOL2 file for binding site location"
    )
    common_parser.add_argument(
        "--dst_dir", type=str, required=True,
        help="Directory for result file saving"
    )
    common_parser.add_argument(
        "--ligand_path", type=str, required=True,
        help="Path to ligand files directory (SDF format)"
    )
    common_parser.add_argument(
        "--smina_path", type=str,
        help="Path to SMINA installation (default: from SMINA environment variable)"
    )
    common_parser.add_argument(
        "--nnscore_path", type=str,
        help="Path to NNScore installation (default: from NNSCORE environment variable)"
    )
    common_parser.add_argument(
        "--mgltool_path", type=str,
        help="Path to MGLTools installation (default: from MGLTOOL environment variable)"
    )
    
    # Train subcommand
    train_parser = subparsers.add_parser(
        "train", parents=[common_parser],
        help="Train a new TB-IEC-Score model"
    )
    train_parser.add_argument(
        "--label_csv", type=str, required=True,
        help="Path to CSV file with ligand labels (name, label)"
    )
    train_parser.add_argument(
        "--model_file", type=str, required=True,
        help="Path to save the trained model"
    )
    train_parser.add_argument(
        "--model_type", type=str, default="xgboost", 
        choices=["xgboost", "svm", "rf"],
        help="Type of model to train (default: xgboost)"
    )
    train_parser.add_argument(
        "--no_hyper_opt", action="store_true",
        help="Disable hyperparameter optimization"
    )
    train_parser.set_defaults(func=train_command)
    
    # Predict subcommand
    predict_parser = subparsers.add_parser(
        "predict", parents=[common_parser],
        help="Make predictions with a trained TB-IEC-Score model"
    )
    predict_parser.add_argument(
        "--model_file", type=str, required=True,
        help="Path to the trained model file"
    )
    predict_parser.set_defaults(func=predict_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute the appropriate command
    args.func(args)


if __name__ == "__main__":
    main() 