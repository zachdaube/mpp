"""
Ensemble XGBoost + GNN - Simple Version
"""
import torch
import numpy as np
import yaml
import sys
import os
from pathlib import Path

# FIXED: Get project root and set paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.data import load_data
from src.models import BaselineModel, GNNModel
from torch_geometric.loader import DataLoader as GeoDataLoader


print("="*80)
print("ENSEMBLE: XGBoost + GNN")
print("="*80)


# ============================================================================
# 1. LOAD/TRAIN XGBOOST
# ============================================================================

print("\n📦 Loading XGBoost...")

# Load config
with open('configs/xgboost_tuned.yaml') as f:
    xgb_config = yaml.safe_load(f)

# Load data (with combined features for XGBoost)
xgb_train, xgb_val, xgb_test = load_data(xgb_config)

# Train XGBoost
print("  🔄 Training XGBoost...")
xgb_model = BaselineModel(xgb_config)
xgb_model.fit(xgb_train, xgb_val)

# Get predictions
xgb_preds = xgb_model.predict(xgb_test)
xgb_y_true = np.array([xgb_test[i]['label'].item() for i in range(len(xgb_test))])

# Evaluate
xgb_mae = np.mean(np.abs(xgb_preds - xgb_y_true))
print(f"  ✅ XGBoost Test MAE: {xgb_mae:.4f}")


# ============================================================================
# 2. LOAD GNN
# ============================================================================

print("\n📦 Loading GNN...")

# Load config
with open('configs/gnn.yaml') as f:
    gnn_config = yaml.safe_load(f)

# Find best checkpoint
checkpoint_dir = Path('results/checkpoints')
gnn_name = gnn_config['name']
checkpoints = list(checkpoint_dir.glob(f"{gnn_name}*.ckpt"))

if not checkpoints:
    print(f"  ❌ No checkpoint found for {gnn_name}")
    print(f"  Please train the model first: python src/train.py --config configs/gnn.yaml")
    sys.exit(1)

# Get best checkpoint (lowest val_mae)
best_ckpt = min(checkpoints, key=lambda p: float(p.stem.split('val_mae=')[1].split('.ckpt')[0]))
print(f"  📦 Loading checkpoint: {best_ckpt.name}")

# Load model
gnn_model = GNNModel.load_from_checkpoint(str(best_ckpt), config=gnn_config)
gnn_model.eval()

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model = gnn_model.to(device)

# Load test data
_, _, gnn_test = load_data(gnn_config)
gnn_loader = GeoDataLoader(gnn_test, batch_size=32, shuffle=False)

# Get predictions
print("  🔄 Getting GNN predictions...")
gnn_preds = []
gnn_y_true = []

with torch.no_grad():
    for batch in gnn_loader:
        batch = batch.to(device)
        pred = gnn_model(batch).squeeze().cpu().numpy()
        gnn_preds.extend(pred if pred.ndim > 0 else [pred.item()])
        gnn_y_true.extend(batch.y.squeeze().cpu().numpy())

gnn_preds = np.array(gnn_preds)
gnn_y_true = np.array(gnn_y_true)

# Evaluate
gnn_mae = np.mean(np.abs(gnn_preds - gnn_y_true))
print(f"  ✅ GNN Test MAE: {gnn_mae:.4f}")


# ============================================================================
# 3. CREATE ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("CREATING ENSEMBLE")
print("="*80)

# Calculate weights (inverse MAE - better models get higher weight)
xgb_weight = 1.0 / xgb_mae
gnn_weight = 1.0 / gnn_mae

# Normalize
total_weight = xgb_weight + gnn_weight
xgb_weight_norm = xgb_weight / total_weight
gnn_weight_norm = gnn_weight / total_weight

print(f"\nModel Weights (inverse MAE):")
print(f"  XGBoost: {xgb_weight_norm:.4f}")
print(f"  GNN:     {gnn_weight_norm:.4f}")

# Create ensemble predictions (weighted average)
ensemble_preds = xgb_weight_norm * xgb_preds + gnn_weight_norm * gnn_preds

# Use ground truth from either model (should be same)
y_true = xgb_y_true

# Calculate ensemble MAE
ensemble_mae = np.mean(np.abs(ensemble_preds - y_true))

# Results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"  XGBoost:  {xgb_mae:.4f} MAE")
print(f"  GNN:      {gnn_mae:.4f} MAE")
print(f"  ENSEMBLE: {ensemble_mae:.4f} MAE  🎯")

# Calculate improvement
best_individual = min(xgb_mae, gnn_mae)
improvement = best_individual - ensemble_mae
improvement_pct = (improvement / best_individual) * 100

print(f"\n💡 Best individual model: {best_individual:.4f} MAE")
print(f"💡 Ensemble improvement:  {improvement:.4f} MAE ({improvement_pct:.2f}%)")

if improvement > 0:
    print("✅ Ensemble is better!")
else:
    print("⚠️  Ensemble didn't improve - models may be too similar")

print("="*80)

# Save results
import pandas as pd
result = {
    'model': 'Ensemble (XGBoost + GNN)',
    'xgboost_mae': xgb_mae,
    'gnn_mae': gnn_mae,
    'ensemble_mae': ensemble_mae,
    'xgboost_weight': xgb_weight_norm,
    'gnn_weight': gnn_weight_norm,
    'improvement': improvement,
    'improvement_pct': improvement_pct
}

Path('results').mkdir(exist_ok=True)
df = pd.DataFrame([result])
df.to_csv('results/ensemble_xgb_gnn.csv', index=False)
print("\n✅ Results saved to results/ensemble_xgb_gnn.csv")
