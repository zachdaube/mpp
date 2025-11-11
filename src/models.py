from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool
import pytorch_lightning as pl
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


#class BaselineModel:
# """Classical ML baseline (RF or XGBoost)"""
# def __init__(self, config):
#     self.config = config
#     model_type = config['model']['type']
    
#     if model_type == 'RandomForest':
#         self.model = RandomForestRegressor(
#             n_estimators=config['model'].get('n_estimators', 100),
#             max_depth=config['model'].get('max_depth', 20),
#             random_state=config['training']['seed'],
#             n_jobs=-1
#         )
#     elif model_type == 'XGBoost':
#         self.model = xgb.XGBRegressor(
#             n_estimators=config['model'].get('n_estimators', 100),
#             max_depth=config['model'].get('max_depth', 6),
#             learning_rate=config['model'].get('learning_rate', 0.1),
#             random_state=config['training']['seed']
#         )
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")

# def fit(self, train_dataset, val_dataset=None):
#     """Train the model"""
#     X_train = np.array([train_dataset[i]['features'].numpy() for i in range(len(train_dataset))])
#     y_train = np.array([train_dataset[i]['label'].item() for i in range(len(train_dataset))])
    
#     self.model.fit(X_train, y_train)

# def predict(self, dataset):
#     """Make predictions"""
#     X = np.array([dataset[i]['features'].numpy() for i in range(len(dataset))])
#     return self.model.predict(X)

# def evaluate(self, dataset):
#     """Calculate MAE"""
#     preds = self.predict(dataset)
#     y_true = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
#     mae = np.mean(np.abs(preds - y_true))
#     return mae

class BaselineModel:
    """Classical ML baseline (RF or XGBoost)"""
    def __init__(self, config):
        self.config = config
        model_type = config['model']['type']
        
        if model_type == 'RandomForest':
            self.model = RandomForestRegressor(
                n_estimators=config['model'].get('n_estimators', 100),
                max_depth=config['model'].get('max_depth', 20),
                min_samples_split=config['model'].get('min_samples_split', 2),
                min_samples_leaf=config['model'].get('min_samples_leaf', 1),
                random_state=config['training']['seed'],
                n_jobs=-1,
                verbose=1  # Show progress
            )
        elif model_type == 'XGBoost':
            self.model = xgb.XGBRegressor(
                n_estimators=config['model'].get('n_estimators', 100),
                max_depth=config['model'].get('max_depth', 6),
                learning_rate=config['model'].get('learning_rate', 0.1),
                subsample=config['model'].get('subsample', 1.0),
                colsample_bytree=config['model'].get('colsample_bytree', 1.0),
                gamma=config['model'].get('gamma', 0),
                reg_alpha=config['model'].get('reg_alpha', 0),
                reg_lambda=config['model'].get('reg_lambda', 1),
                random_state=config['training']['seed'],
                verbosity=1  # Show progress
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, train_dataset, val_dataset=None):
        """Train the model"""
        X_train = np.array([train_dataset[i]['features'].numpy() for i in range(len(train_dataset))])
        y_train = np.array([train_dataset[i]['label'].item() for i in range(len(train_dataset))])
        
        # For XGBoost, can use validation set for early stopping
        if isinstance(self.model, xgb.XGBRegressor) and val_dataset is not None:
            X_val = np.array([val_dataset[i]['features'].numpy() for i in range(len(val_dataset))])
            y_val = np.array([val_dataset[i]['label'].item() for i in range(len(val_dataset))])
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
    
    def predict(self, dataset):
        """Make predictions"""
        X = np.array([dataset[i]['features'].numpy() for i in range(len(dataset))])
        return self.model.predict(X)
    
    def evaluate(self, dataset):
        """Calculate MAE"""
        preds = self.predict(dataset)
        y_true = np.array([dataset[i]['label'].item() for i in range(len(dataset))])
        mae = np.mean(np.abs(preds - y_true))
        return mae

class GNNModel(pl.LightningModule):
    """Graph Neural Network with edge features for molecular property prediction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        hidden_dim = config['model'].get('hidden_dim', 128)
        num_layers = config['model'].get('num_layers', 3)
        dropout = config['model'].get('dropout', 0.2)
        
        # OGB atom features: 9 different feature types
        atom_feature_dims = get_atom_feature_dims()
        
        # OGB bond features: 3 different feature types
        bond_feature_dims = get_bond_feature_dims()
        
        # Atom encoder
        self.atom_encoders = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in atom_feature_dims
        ])
        
        # Bond encoder - encode edge features
        self.bond_encoders = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in bond_feature_dims
        ])
        
        # Edge network for NNConv (processes edge features)
        edge_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
        )
        
        # GNN layers with edge features
        self.convs = nn.ModuleList()
        self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
        
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Prediction head
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode atom features
        h = 0
        for i, encoder in enumerate(self.atom_encoders):
            h = h + encoder(x[:, i])
        
        # Encode edge features (bond types, stereochemistry, conjugation)
        edge_h = 0
        for i, encoder in enumerate(self.bond_encoders):
            edge_h = edge_h + encoder(edge_attr[:, i])
        
        # Message passing WITH edge features
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index, edge_h)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        h = global_mean_pool(h, batch)
        
        # Prediction
        return self.fc(h)
    
    def training_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        loss = F.l1_loss(pred, batch.y.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch.y))
        self.log('train_mae', loss, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        mae = F.l1_loss(pred, batch.y.squeeze())
        self.log('val_loss', mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch.y))
        self.log('val_mae', mae, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return mae
    
    def test_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        mae = F.l1_loss(pred, batch.y.squeeze())
        self.log('test_mae', mae, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return mae
    
    def configure_optimizers(self):
        lr = self.config['training'].get('lr', 0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_mae'}
        }


class GATModel(pl.LightningModule):
    """Graph Attention Network for molecular property prediction"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        hidden_dim = config['model'].get('hidden_dim', 128)
        num_layers = config['model'].get('num_layers', 3)
        dropout = config['model'].get('dropout', 0.3)
        heads = config['model'].get('heads', 4)
        
        # OGB features
        atom_feature_dims = get_atom_feature_dims()
        
        # Atom encoder
        self.atom_encoders = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in atom_feature_dims
        ])
        
        # GAT layers with multi-head attention
        self.convs = nn.ModuleList()
        
        # First layer: hidden_dim -> hidden_dim (with heads)
        self.convs.append(GATConv(
            hidden_dim, 
            hidden_dim // heads, 
            heads=heads, 
            dropout=dropout,
            add_self_loops=True,
            concat=True
        ))
        
        # Middle layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                concat=True
            ))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Prediction head
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode atom features
        h = 0
        for i, encoder in enumerate(self.atom_encoders):
            h = h + encoder(x[:, i])
        
        # Message passing with attention
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        h = global_mean_pool(h, batch)
        
        # Prediction
        return self.fc(h)
    
    def training_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        loss = F.l1_loss(pred, batch.y.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch.y))
        self.log('train_mae', loss, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        mae = F.l1_loss(pred, batch.y.squeeze())
        self.log('val_loss', mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch.y))
        self.log('val_mae', mae, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return mae
    
    def test_step(self, batch, batch_idx):
        pred = self(batch).squeeze()
        mae = F.l1_loss(pred, batch.y.squeeze())
        self.log('test_mae', mae, on_step=False, on_epoch=True, batch_size=len(batch.y))
        return mae
    
    def configure_optimizers(self):
        lr = self.config['training'].get('lr', 0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_mae'}
        }