# Molecular Property Prediction: Lipophilicity Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning comparison study for predicting molecular lipophilicity from SMILES strings. Achieves **0.5283 test MAE** using an ensemble of XGBoost and Graph Convolutional Networks.

**Author:** Zach Daube  
**Institution:** Emory University
**Date:** November 2025

---

## 🎯 Project Overview

This project implements and compares three paradigms for molecular property prediction:
- **Classical ML** (Random Forest, XGBoost) with hand-crafted features
- **Graph Neural Networks** (GCN, GAT) learning from molecular graphs
- **Transformers** (ChemBERTa) processing SMILES sequences

### Key Results

| Architecture | Test MAE | Parameters | Training Time |
|-------------|----------|------------|---------------|
| XGBoost (Combined Features) | 0.5588 | ~100K | 30s |
| **GCN Medium** | **0.5502** | 73K | 3 min |
| ChemBERT (Fine-tuned) | 0.6939 | 45M | 4 min |
| **🏆 Ensemble (XGBoost + GCN)** | **0.5283** | ~173K | 3.5 min |

**30.7% improvement** over baseline Random Forest (0.762 → 0.528 MAE)

---

## 📁 Repository Structure
```
molecular-property-prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── REPORT.md                          # Detailed technical report
├── data/                              # Dataset files
│   ├── lipophilicity_train.csv
│   ├── lipophilicity_val.csv
│   └── lipophilicity_test.csv
├── src/                               # Source code
│   ├── __init__.py
│   ├── train.py                       # Main training script
│   ├── data.py                        # Data loading and featurization
│   └── models.py                      # Model architectures
├── configs/                           # Model configurations
│   ├── baseline.yaml
│   ├── baseline_rf_combined.yaml
│   ├── xgboost_tuned.yaml
│   ├── gnn.yaml                       # Best GCN config
│   ├── gnn_small.yaml
│   ├── gnn_large.yaml
│   ├── gat.yaml
│   ├── chemberta.yaml
│   └── chemberta_frozen.yaml
├── scripts/                           # Utility scripts
│   ├── ensemble.py                    # Ensemble XGBoost + GCN
│   └── run_all_experiments.py        # Batch training
├── results/                           # Outputs (not in repo)
│   ├── results.csv
│   ├── checkpoints/
│   └── models/
├── plots/                             # Visualizations
│   ├── model_comparison.png
│   ├── architecture_comparison.png
│   ├── ensemble_analysis.png
│   └── ...
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- 8GB RAM minimum
- 2GB disk space

### Installation
```bash
# Clone repository
git clone https://github.com/zachdaube/mpp.git
cd mpp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import rdkit; print('✅ Setup complete!')"
```

### Download Data

The lipophilicity dataset is included in the `data/` folder.

---

## 🏃 Running Experiments

### 1. Train Individual Models

**Classical ML (XGBoost):**
```bash
python src/train.py --config configs/xgboost_tuned.yaml
```

**Graph Neural Network:**
```bash
python src/train.py --config configs/gnn.yaml
```

**Transformer:**
```bash
python src/train.py --config configs/chemberta.yaml
```

### 2. Train All Models (Batch)
```bash
python scripts/run_all_experiments.py
```

This runs all configurations sequentially (~30-40 minutes on GPU).

### 3. Create Ensemble
```bash
python scripts/ensemble.py
```

Combines XGBoost and GCN for best performance (0.5283 MAE).

### 4. Generate Visualizations
```bash
python scripts/generate_plots.py
```

Creates all figures in `plots/` directory.

---

## 📊 Reproducing Results

### Expected Outputs

After running all experiments, you should see:
```bash
results/results.csv:
    baseline_rf:           0.7620 MAE
    xgboost_tuned:         0.5588 MAE
    gnn_gcn:               0.5502 MAE
    chemberta_finetune:    0.6939 MAE
    ensemble:              0.5283 MAE ✅
```

### Training Times (NVIDIA T4 GPU)

- Classical ML: 30-60 seconds per model
- GNN: 3-5 minutes per model
- Transformer: 3-4 minutes per model
- Total: ~40 minutes for all models

### Checkpoints

Trained models are saved in:
- GNN checkpoints: `results/checkpoints/gnn_*.ckpt`
- Baseline models: `results/models/*.pkl`

---

## 🔬 Key Findings

### 1. Feature Engineering is Critical

Combined features (Morgan + RDKit + 42 descriptors) improved classical ML significantly
- Simple fingerprints: 0.762 MAE
- Combined features: 0.617 MAE (-13.9%)
- XGBoost + Combined: 0.559 MAE (-8.7%)

### 2. GNNs Outperform Transformers on Small Datasets

| Model | Parameters | Samples/Param | Test MAE | Generalization |
|-------|------------|---------------|----------|----------------|
| GCN | 73K | 0.040 | **0.550** | ✅ Excellent |
| ChemBERT | 45M | 0.000066 | 0.694 | ❌ Severe overfit |

**Insight:** GNNs' graph-based inductive bias provides crucial regularization for small molecular datasets (2,940 samples).

### 3. Ensemble Achieves Best Performance

Weighted averaging of complementary models:
- XGBoost captures explicit feature relationships
- GCN learns implicit graph topology
- **5.5% improvement** over best individual model

### 4. GAT Fails Due to Model Complexity

GAT (with attention) severely overfit:
- Val MAE: 0.647
- Test MAE: 0.722 (gap: +0.075)

Attention mechanisms add capacity that small datasets cannot support.

---

## 📈 Model Configurations

### Best Hyperparameters

**XGBoost (Combined Features):**
```yaml
n_estimators: 500
max_depth: 8
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
reg_lambda: 1.0
features: 4,138 (Morgan + RDKit + Descriptors)
```

**GCN Medium:**
```yaml
hidden_dim: 128
num_layers: 3
dropout: 0.2
learning_rate: 0.001
batch_size: 32
parameters: 72,705
```

**Ensemble:**
```yaml
models: [GCN, XGBoost]
weights: [50.1%, 49.9%]
strategy: Weighted average (inverse MAE)
```

---

## 🛠️ Development

### Project Structure

**Core Files:**
- `src/train.py`: Main training loop with PyTorch Lightning
- `src/data.py`: Dataset classes for fingerprints, graphs, and SMILES
- `src/models.py`: Model architectures (BaselineModel, GNNModel, ChemBERTaModel)

**Configuration:**
- YAML-based configs in `configs/`
- Override any parameter via command line

**Logging:**
- Weights & Biases integration for experiment tracking
- Results saved to `results/results.csv`

### Adding New Models

1. Create model class in `src/models.py`
2. Add training logic in `src/train.py`
3. Create config in `configs/your_model.yaml`
4. Run: `python src/train.py --config configs/your_model.yaml`

---


## 🙏 Acknowledgments

- **Datasets:** MoleculeNet (lipophilicity)
- **Frameworks:** PyTorch, PyTorch Geometric, RDKit, XGBoost
- **Pretrained Models:** ChemBERTa (Hugging Face)
- **Visualization:** Weights & Biases, Matplotlib, Seaborn

---

## 📧 Contact

**Zach Daube**  
Emory University
Email: zach.daube@emory.edu  
GitHub: [@zachdaube](https://github.com/zachdaube)

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in configs
batch_size: 16  # instead of 32
```

**2. RDKit Import Error**
```bash
conda install -c conda-forge rdkit
```

**3. PyTorch Geometric Installation**
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**4. WandB Login**
```bash
wandb login
# Enter your API key from wandb.ai/authorize
```

---

## 📖 Additional Resources

- [Full Technical Report](REPORT.md)
- [WandB Project Dashboard](https://wandb.ai/zachdaube-emory-university/molecular-property-prediction)

---

**Last Updated:** November 2025  
**Version:** 1.0.0
