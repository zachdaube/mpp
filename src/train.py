import yaml
import argparse
import random
import numpy as np
import torch
from pathlib import Path
import sys
import pandas as pd
import time
import wandb
from torch_geometric.loader import DataLoader as GeoDataLoader
import os
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Also change working directory
from src.models import BaselineModel, GNNModel, GATModel, ChemBERTaModel
from src.data import load_data
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For PyTorch Lightning
    pl.seed_everything(seed, workers=True)


def train_baseline_model(config, train_dataset, val_dataset, test_dataset, wandb_run):
    """Train classical ML models (RF, XGBoost)"""
    model_type = config['model']['type']
    print(f"\nTraining {model_type} model...")
    
    model = BaselineModel(config)
    
    start_time = time.time()
    model.fit(train_dataset, val_dataset)
    train_time = time.time() - start_time
    
    # Evaluate
    train_mae = model.evaluate(train_dataset)
    val_mae = model.evaluate(val_dataset)
    test_mae = model.evaluate(test_dataset)
    
    num_params = 0  # Classical models don't have traditional params
    
    return {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'train_time': train_time,
        'num_params': num_params
    }


def train_gnn_model(config, train_dataset, val_dataset, test_dataset, wandb_run):
    """Train GNN models"""
    print(f"\nTraining GNN model...")
    
    # Create data loaders
    batch_size = config['training'].get('batch_size', 32)
    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=batch_size)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model_type = config['model']['type']
    if model_type == 'GNN':
        model = GNNModel(config)
    elif model_type == 'GAT':
        model = GATModel(config)
    else:
        raise ValueError(f"Unknown GNN type: {model_type}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # WandB logger for PyTorch Lightning
    wandb_logger = WandbLogger(
        experiment=wandb_run,  # Reuse existing run
        log_model=False
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='results/checkpoints',
        filename=f'{config["name"]}-{{epoch:02d}}-{{val_mae:.4f}}',
        monitor='val_mae',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=config['training'].get('patience', 20),
        mode='min',
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training'].get('max_epochs', 100),
        callbacks=[checkpoint_callback, early_stop],
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accelerator='auto',  # Automatically use GPU if available
        devices=1,
        deterministic=True  # For reproducibility
    )
    
    # Train
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_time = time.time() - start_time
    
    # Evaluate on best checkpoint
    trainer.test(model, train_loader, verbose=False, ckpt_path='best')
    train_results = trainer.callback_metrics
    
    trainer.test(model, val_loader, verbose=False, ckpt_path='best')
    val_results = trainer.callback_metrics
    
    trainer.test(model, test_loader, verbose=False, ckpt_path='best')
    test_results = trainer.callback_metrics
    
    return {
        'train_mae': train_results.get('test_mae', float('nan')),
        'val_mae': val_results.get('test_mae', float('nan')),
        'test_mae': test_results.get('test_mae', float('nan')),
        'train_time': train_time,
        'num_params': num_params
    }



def train_chemberta_model(config, train_dataset, val_dataset, test_dataset, wandb_run):
    """Train ChemBERTa models"""
    print(f"\nTraining ChemBERTa model...")
    
    # Create data loaders (regular DataLoader, not GeoDataLoader)
    batch_size = config['training'].get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Calculate training steps for scheduler
    num_training_steps = len(train_loader) * config['training'].get('max_epochs', 10)
    config['training']['num_training_steps'] = num_training_steps
    
    # Create model
    model = ChemBERTaModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters ({trainable_params:,} trainable)")
    
    # WandB logger
    wandb_logger = WandbLogger(experiment=wandb_run, log_model=False)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='results/checkpoints',
        filename=f'{config["name"]}-{{epoch:02d}}-{{val_mae:.4f}}',
        monitor='val_mae',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    
    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=config['training'].get('patience', 5),  # Lower patience for transformers
        mode='min',
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training'].get('max_epochs', 10),
        callbacks=[checkpoint_callback, early_stop],
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accelerator='auto',
        devices=1,
        deterministic=True,
        gradient_clip_val=1.0  # Clip gradients for stability
    )
    
    # Train
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_time = time.time() - start_time
    
    # Evaluate
    trainer.test(model, train_loader, verbose=False, ckpt_path='best')
    train_results = trainer.callback_metrics
    
    trainer.test(model, val_loader, verbose=False, ckpt_path='best')
    val_results = trainer.callback_metrics
    
    trainer.test(model, test_loader, verbose=False, ckpt_path='best')
    test_results = trainer.callback_metrics
    
    return {
        'train_mae': train_results.get('test_mae', float('nan')),
        'val_mae': val_results.get('test_mae', float('nan')),
        'test_mae': test_results.get('test_mae', float('nan')),
        'train_time': train_time,
        'num_params': trainable_params  # Report only trainable params
    }


def train_model(config):
    """Main training function - orchestrates everything"""
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"Model Type: {config['model']['type']}")
    print(f"{'='*60}\n")
    
    # Initialize WandB
    run = wandb.init(
        project="molecular-property-prediction",
        name=config['name'],
        config=config,
        reinit=True,
        tags=[config['model']['type'], config['data'].get('featurizer', 'unknown')]
    )
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Load data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(config)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Log dataset info
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/val_size": len(val_dataset),
        "dataset/test_size": len(test_dataset)
    })
    
    model_type = config['model']['type']
    
    # Route to appropriate training function
    if model_type in ['RandomForest', 'XGBoost']:
        metrics = train_baseline_model(config, train_dataset, val_dataset, test_dataset, run)
    elif model_type in ['GNN', 'GAT']:
        metrics = train_gnn_model(config, train_dataset, val_dataset, test_dataset, run)
    elif model_type == 'ChemBERTa':  # NEW
        metrics = train_chemberta_model(config, train_dataset, val_dataset, test_dataset, run)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Final Results for {config['name']}")
    print(f"{'='*60}")
    print(f"  Train MAE: {metrics['train_mae']:.4f}")
    print(f"  Val MAE:   {metrics['val_mae']:.4f}")
    print(f"  Test MAE:  {metrics['test_mae']:.4f}")
    print(f"  Train time: {metrics['train_time']:.2f}s ({metrics['train_time']/60:.2f} min)")
    print(f"  Num params: {metrics['num_params']:,}")
    print(f"{'='*60}\n")
    
    # Log final metrics to WandB
    wandb.log({
        "final/train_mae": metrics['train_mae'],
        "final/val_mae": metrics['val_mae'],
        "final/test_mae": metrics['test_mae'],
        "final/train_time": metrics['train_time'],
        "final/num_params": metrics['num_params']
    })
    
    # Create summary for WandB
    wandb.run.summary.update({
        "train_mae": metrics['train_mae'],
        "val_mae": metrics['val_mae'],
        "test_mae": metrics['test_mae'],
        "train_time": metrics['train_time'],
        "num_params": metrics['num_params'],
        "model_type": model_type,
        "featurizer": config['data'].get('featurizer', 'unknown')
    })
    
    wandb.finish()
    
    # Save to CSV
    results = {
        'name': config['name'],
        'model_type': model_type,
        'featurizer': config['data'].get('featurizer', 'unknown'),
        'train_mae': metrics['train_mae'],
        'val_mae': metrics['val_mae'],
        'test_mae': metrics['test_mae'],
        'train_time': metrics['train_time'],
        'num_params': metrics['num_params'],
        'seed': config['training']['seed'],
        'hidden_dim': config['model'].get('hidden_dim', 'N/A'),
        'num_layers': config['model'].get('num_layers', 'N/A'),
        'n_estimators': config['model'].get('n_estimators', 'N/A')
    }
    
    results_file = Path('results/results.csv')
    results_file.parent.mkdir(exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(results_file, mode='a', header=not results_file.exists(), index=False)
    
    print(f"✅ Results saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train molecular property prediction models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--sweep', action='store_true', help='Run as part of WandB sweep')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Train
    results = train_model(config)
    
    return results


if __name__ == '__main__':
    main()
