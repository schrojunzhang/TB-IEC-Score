# TB-IEC-Score: An accurate and efficient machine learning-based scoring function for virtual screening

## Overview
TB-IEC-Score is a machine learning-based scoring function that integrates multiple energy calculation terms from various scoring methods to improve virtual screening accuracy in drug discovery.

## Project Structure
```
TB-IEC-Score/
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── setup.py                   # Package setup script
├── tb_iecs/                   # Main package
│   ├── __init__.py            # Package initialization
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── model.py           # Model implementation (XGBoost, SVM, RF)
│   │   └── pipeline.py        # Main pipeline implementation
│   ├── descriptors/           # Feature extraction modules
│   │   ├── __init__.py
│   │   ├── smina.py           # SMINA energy terms
│   │   ├── nnscore.py         # NNScore 2.0 energy terms
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── data.py            # Dataset handling
│       ├── metrics.py         # Evaluation metrics
│       └── feature_importance.py # Feature importance calculation
├── cli.py                     # Command-line interface
├── experiments/               # Experimental code
│   ├── svm_model.py           # SVM model experiments
│   └── algo_compare.py        # Algorithm comparison
└── examples/                  # Example usage
    ├── training_example.py
    └── prediction_example.py
```

## Installation
1. Clone the repository:
```
cd $HOME
git clone https://github.com/username/TB-IEC-Score.git
cd TB-IEC-Score
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install external tools:

### Install smina:
```
cd $HOME/TB-IEC-Score/libs/Smina
chmod +x ./smina
export SMINA=$HOME/TB-IEC-Score/libs/Smina
```

### Install nnscore 2.0
```
# Set environment variable: 
export NNSCORE=$HOME/TB-IEC-Score/libs/NNscore
```

### Install mgltool
```
cd $HOME
download mgltool: https://ccsb.scripps.edu/mgltools/download/491/
tar -zxvf mgltools_x86_64Linux2_1.5.7p1.tar.gz
cd mgltools_x86_64Linux2_1.5.7/
chmod +x install.sh
./install.sh
export MGLTOOL=/opt/mgltools/1.5.7
```


## Quick Start

### Using the CLI

```bash
# Training mode
python -m tb_iecs train --protein_file path/to/protein.pdb --ligand_path path/to/ligands \
                       --label_csv path/to/labels.csv --crystal_ligand_file path/to/crystal.mol2 \
                       --model_file path/to/model.pkl --dst_dir path/to/results

# Testing/prediction mode
python -m tb_iecs predict --protein_file path/to/protein.pdb --ligand_path path/to/ligands \
                        --crystal_ligand_file path/to/crystal.mol2 --model_file path/to/model.pkl \
                        --dst_dir path/to/results
```

### Using the Python API

```python
from tb_iecs.core.pipeline import TBIECPipeline

# Initialize pipeline
pipeline = TBIECPipeline(
    protein_file="path/to/protein.pdb",
    crystal_ligand_file="path/to/crystal.mol2",
    dst_dir="path/to/results"
)

# Training
pipeline.train(
    ligand_path="path/to/ligands",
    label_csv="path/to/labels.csv",
    model_file="path/to/model.pkl"
)

# Prediction
results = pipeline.predict(
    ligand_path="path/to/ligands",
    model_file="path/to/model.pkl"
)
```

## Input Format

- **protein_file**: Protein structure in PDB format
- **crystal_ligand_file**: Crystal ligand in MOL2 format for binding site location
- **label_csv**: CSV file with columns 'name' and 'label' (1 for active, 0 for inactive)
- **ligand_path**: Directory containing ligand files in SDF format

## Output

- **model_file**: Trained model saved as a pickle file
- **result.csv**: Prediction results with predicted labels and probabilities

## Advanced Usage

See the `examples/` directory for more detailed usage examples.

## Citation

If you use TB-IEC-Score in your research, please cite:
```
Zhang, X., Shen, C., Jiang, D. et al. TB-IECS: an accurate machine learning-based scoring function for virtual screening. J Cheminform 15, 63 (2023). https://doi.org/10.1186/s13321-023-00731-x
```
