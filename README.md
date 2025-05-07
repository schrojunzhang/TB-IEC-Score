# TB-IEC-Score

## Overview

TB-IEC-Score (Tuberculosis-Integrated Energy Calculation Score) is a machine learning-based scoring function that integrates multiple energy calculation terms from various scoring methods to improve virtual screening accuracy in drug discovery. This method combines energy terms from scoring functions like SMINA and NNScore 2.0 as features, using machine learning models such as XGBoost, SVM, or Random Forest for prediction.

[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs13321--023--00731--x-blue)](https://doi.org/10.1186/s13321-023-00731-x)

## Key Features

- **High Accuracy**: Significantly improves virtual screening prediction accuracy by integrating energy terms from multiple scoring functions
- **High Efficiency**: Optimized parallel processing pipeline supporting rapid screening of large-scale compounds
- **Flexibility**: Supports multiple machine learning models (XGBoost, SVM, Random Forest)
- **Usability**: Provides a simple command-line interface and Python API

## Installation Guide

### Prerequisites

- Python 3.6+
- SMINA
- MGLTools

### Step 1: Clone the Repository

```bash
git clone https://github.com/username/TB-IEC-Score.git
cd TB-IEC-Score
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install External Tools

#### Install SMINA

```bash
cd $HOME/TB-IEC-Score/libs/Smina
chmod +x ./smina
chmod +x ./vina
```

#### Install MGLTools

```bash
cd $HOME/TB-IEC-Score/libs
# Download MGLTools (if not already downloaded)
wget https://ccsb.scripps.edu/mgltools/download/491/mgltools_x86_64Linux2_1.5.7p1.tar.gz
tar -zxvf mgltools_x86_64Linux2_1.5.7p1.tar.gz
rm mgltools_x86_64Linux2_1.5.7p1.tar.gz
cd mgltools_x86_64Linux2_1.5.7/
chmod +x install.sh
./install.sh
```

## Project Structure

```
TB-IEC-Score/
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── tb_iecs/               # Main package
│   ├── core/              # Core functionality
│   │   ├── model.py       # Model implementation (XGBoost, SVM, RF)
│   │   └── pipeline.py    # Main pipeline implementation
│   ├── descriptors/       # Feature extraction modules
│   │   ├── smina.py       # SMINA energy terms
│   │   ├── nnscore.py     # NNScore 2.0 energy terms
│   └── utils/             # Utility functions
│       ├── data.py        # Dataset handling
│       ├── metrics.py     # Evaluation metrics
│       └── feature_importance.py # Feature importance calculation
└── examples/              # Usage examples
    ├── training_example.py   # Training example
    └── prediction_example.py # Prediction example
```

## Quick Start

### Command Line Interface

#### Training Mode

```bash
python -m tb_iecs train --protein_file path/to/protein.pdb \
                       --ligand_path path/to/ligands \
                       --label_csv path/to/labels.csv \
                       --crystal_ligand_file path/to/crystal.mol2 \
                       --model_file path/to/model.pkl \
                       --dst_dir path/to/results \
                       --model_type xgboost \
                       --num_workers 4
```

#### Prediction Mode

```bash
python -m tb_iecs predict --protein_file path/to/protein.pdb \
                        --ligand_path path/to/ligands \
                        --crystal_ligand_file path/to/crystal.mol2 \
                        --model_file path/to/model.pkl \
                        --dst_dir path/to/results \
                        --num_workers 4
```

### Python API

#### Training Example

```python
from tb_iecs.core.pipeline import TBIECPipeline

# Initialize pipeline
pipeline = TBIECPipeline(
    protein_file="path/to/protein.pdb",
    crystal_ligand_file="path/to/crystal.mol2",
    dst_dir="path/to/results",
    num_workers=4
)

# Training
metrics = pipeline.train(
    ligand_path="path/to/ligands",
    label_csv="path/to/labels.csv",
    model_file="path/to/model.pkl",
    model_type="xgboost",
    hyper_opt=True
)

# Print evaluation metrics
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

#### Prediction Example

```python
from tb_iecs.core.pipeline import TBIECPipeline

# Initialize pipeline
pipeline = TBIECPipeline(
    protein_file="path/to/protein.pdb",
    crystal_ligand_file="path/to/crystal.mol2",
    dst_dir="path/to/results",
    num_workers=4
)

# Prediction
results = pipeline.predict(
    ligand_path="path/to/ligands",
    model_file="path/to/model.pkl",
    output_csv="path/to/results/result.csv"
)

# View Top-10 prediction results
top_compounds = results.sort_values(by='pred_y_proba', ascending=False).head(10)
for i, (_, row) in enumerate(top_compounds.iterrows(), 1):
    status = "ACTIVE" if row['pred_y'] == 1 else "INACTIVE"
    print(f"{i}. {row['name']} - Score: {row['pred_y_proba']:.4f} - Prediction: {status}")
```

## Input Format

TB-IEC-Score accepts the following input formats:

- **protein_file**: Protein structure in PDB format
- **crystal_ligand_file**: Crystal ligand in MOL2 format for binding site location
- **label_csv**: CSV file with columns 'name' and 'label' (1 for active, 0 for inactive)
- **ligand_path**: Directory containing ligand files in MOL2 format

## Output

- **model_file**: Trained model saved in pickle format
- **result.csv**: Prediction results with predicted labels and probabilities

## Parameter Description

### Main Parameters

| Parameter         | Description                           | Default   |
|-------------------|---------------------------------------|-----------|
| protein_file      | Path to protein PDB file              | Required  |
| crystal_ligand_file | Path to crystal ligand MOL2 file    | Required  |
| ligand_path       | Directory containing ligand files     | Required  |
| label_csv         | CSV file with ligand labels           | Required for training |
| model_file        | Path to model file                    | Required  |
| dst_dir           | Directory for results                 | ./results |
| model_type        | Model type (xgboost, svm, rf)         | xgboost   |
| num_workers       | Number of worker threads for parallel processing | 1 |
| hyper_opt         | Whether to optimize hyperparameters   | True      |

## Example Usage

See the `examples/` directory for more detailed usage examples.


## Citation

If you use TB-IEC-Score in your research, please cite:

```
Zhang, X., Shen, C., Jiang, D. et al. TB-IECS: an accurate machine learning-based scoring function for virtual screening. J Cheminform 15, 63 (2023). https://doi.org/10.1186/s13321-023-00731-x
```

## License

[Please specify appropriate license]

## Contributing

We welcome issue reports and pull requests. For major changes, please open an issue first to discuss what you would like to change.
