import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset as GeoDataset
from ogb.utils import smiles2graph



class MoleculeDataset(Dataset):
    """Dataset for fingerprint-based models (RF, XGBoost)"""
    def __init__(self, smiles_list, labels, featurizer='fingerprint', fp_size=2048, radius=2):
        self.smiles = smiles_list
        self.labels = labels
        self.featurizer_type = featurizer
        self.fp_size = fp_size
        self.radius = radius
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        label = self.labels[idx]
        
        if self.featurizer_type == 'fingerprint':
            features = self._smiles_to_fingerprint(smiles)
        elif self.featurizer_type == 'combined':
            features = self._smiles_to_combined_features(smiles)
        else:
            raise ValueError(f"Unknown featurizer: {self.featurizer_type}")
        
        return {
            'features': torch.FloatTensor(features),
            'label': torch.FloatTensor([label]),
            'smiles': smiles
        }
    
    
    def _smiles_to_fingerprint(self, smiles):
        """Convert SMILES to Morgan fingerprint"""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.fp_size)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fp_size)
        fp = gen.GetFingerprint(mol)
        return np.array(fp)
    
    def _smiles_to_combined_features(self, smiles):
        """Combined features: Morgan + RDKit + Descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(4106)  # 2048 + 2048 + 10
        
        # 1. Morgan fingerprint
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
        morgan = gen.GetFingerprint(mol)
        morgan_array = np.array(morgan)
        
        # 2. RDKit fingerprint (topological paths)
        rdkit_fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=2048)
        rdkit_array = np.array(rdkit_fp)
        
        # 3. Molecular descriptors
        try:
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.NumSaturatedRings(mol),
            ]
        except:
            descriptors = [0] * 10
        
        # Combine all
        return np.concatenate([morgan_array, rdkit_array, np.array(descriptors)])
    

def mol_to_graph(smiles):
    """Convert SMILES to PyTorch Geometric graph using OGB"""
    try:
        graph_dict = smiles2graph(smiles)
        
        data = Data(
            x=torch.tensor(graph_dict['node_feat'], dtype=torch.long),
            edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph_dict['edge_feat'], dtype=torch.long)
        )
        return data
    except:
        # Handle invalid SMILES
        return None


class GraphDataset(GeoDataset):
    """Dataset for graph-based models (GNN)"""
    def __init__(self, smiles_list, labels):
        super().__init__()
        self.graphs = []
        self.labels = []
        
        for smi, label in zip(smiles_list, labels):
            graph = mol_to_graph(smi)
            if graph is not None:
                graph.y = torch.tensor([label], dtype=torch.float)
                self.graphs.append(graph)
                self.labels.append(label)
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx]
    
class SMILESDataset(Dataset):
    """Dataset for sequence-based models (ChemBERTa, transformers)"""
    def __init__(self, smiles_list, labels):
        self.smiles = smiles_list
        self.labels = labels
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'label': torch.FloatTensor([self.labels[idx]])
        }
    
    
def load_data(config):
    """Load train/val/test datasets"""
    train_df = pd.read_csv('data/lipophilicity_train.csv')
    val_df = pd.read_csv('data/lipophilicity_val.csv')
    test_df = pd.read_csv('data/lipophilicity_test.csv')
    
    featurizer = config['data']['featurizer']
    
    if featurizer in ['fingerprint', 'combined']:
        train_dataset = MoleculeDataset(
            train_df['Drug'].values,
            train_df['Y'].values,
            featurizer=featurizer,
            fp_size=config['data'].get('fp_size', 2048),
            radius=config['data'].get('radius', 2)
        )
        val_dataset = MoleculeDataset(
            val_df['Drug'].values,
            val_df['Y'].values,
            featurizer=featurizer,
            fp_size=config['data'].get('fp_size', 2048),
            radius=config['data'].get('radius', 2)
        )
        test_dataset = MoleculeDataset(
            test_df['Drug'].values,
            test_df['Y'].values,
            featurizer=featurizer,
            fp_size=config['data'].get('fp_size', 2048),
            radius=config['data'].get('radius', 2)
        )
    
    elif featurizer == 'graph':
        train_dataset = GraphDataset(
            train_df['Drug'].values,
            train_df['Y'].values
        )
        val_dataset = GraphDataset(
            val_df['Drug'].values,
            val_df['Y'].values
        )
        test_dataset = GraphDataset(
            test_df['Drug'].values,
            test_df['Y'].values
        )

    elif featurizer == 'smiles_text': 
        train_dataset = SMILESDataset(
            train_df['Drug'].values,
            train_df['Y'].values
        )
        val_dataset = SMILESDataset(
            val_df['Drug'].values,
            val_df['Y'].values
        )
        test_dataset = SMILESDataset(
            test_df['Drug'].values,
            test_df['Y'].values
        )
    else:
        raise ValueError(f"Unknown featurizer: {featurizer}")
    
    return train_dataset, val_dataset, test_dataset
