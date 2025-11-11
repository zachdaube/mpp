from rdkit import Chem
import pandas as pd

def sanity_check(csv_path: str, smiles_col: str = "Drug") -> pd.DataFrame:
    """Load a CSV and verify all SMILES are valid RDKit molecules.
    Returns a cleaned DataFrame with only valid entries."""
    df = pd.read_csv(csv_path)
    valid, invalid = 0, 0

    def check(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            nonlocal invalid
            invalid += 1
            return None
        else:
            nonlocal valid
            valid += 1
            return Chem.MolToSmiles(mol, canonical=True)  # canonicalize only for consistency

    df["smiles_checked"] = df[smiles_col].apply(check)
    print(f"{csv_path}: {valid} valid, {invalid} invalid")
    return df[df["smiles_checked"].notna()].reset_index(drop=True)

def main():

    # --- Run it on your splits ---
    train_df = sanity_check('data/lipophilicity_train.csv')
    val_df = sanity_check('data/lipophilicity_val.csv')
    test_df = sanity_check('data/lipophilicity_test.csv')

if __name__ == "__main__":
    main()