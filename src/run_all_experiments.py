"""
Run all experiments systematically
"""
import subprocess
import sys
from pathlib import Path
import time
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(config_path):
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting: {config_path}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', 'src/train.py', '--config', config_path],
            check=True,
            capture_output=False  # Show output in real-time
        )
        elapsed = time.time() - start_time
        print(f"\n✅ Completed {config_path} in {elapsed/60:.2f} minutes\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {config_path}")
        print(f"Error: {e}\n")
        return False


def main():
    # Define all experiments
    experiments = [
        # Classical ML baselines
        'configs/baseline.yaml',
        'configs/baseline_improved.yaml',
        'configs/baseline_combined.yaml',
        'configs/xgboost.yaml',
        'configs/xgboost_tuned.yaml',
        
        # GNN variants
        'configs/gnn_small.yaml',
        'configs/gnn.yaml',
        'configs/gnn_large.yaml',
    ]
    
    # Filter to only configs that exist
    existing_experiments = []
    for exp in experiments:
        if Path(exp).exists():
            existing_experiments.append(exp)
        else:
            print(f"⚠️  Skipping {exp} (file not found)")
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT QUEUE: {len(existing_experiments)} experiments")
    print(f"{'='*70}")
    for i, exp in enumerate(existing_experiments, 1):
        print(f"{i}. {exp}")
    print(f"{'='*70}\n")
    
    # Run all experiments
    results = []
    start_time = time.time()
    
    for i, config_path in enumerate(existing_experiments, 1):
        print(f"\n[{i}/{len(existing_experiments)}] Running experiment...")
        success = run_experiment(config_path)
        results.append((config_path, success))
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Completed: {sum(1 for _, s in results if s)}/{len(results)}")
    print(f"\nResults:")
    for config, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {config}")
    print(f"{'='*70}\n")
    
    # Load and display final results
    try:
        import pandas as pd
        df = pd.read_csv('results/results.csv')
        print("\n📊 FINAL RESULTS (sorted by test MAE):")
        print("="*70)
        df_sorted = df.sort_values('test_mae')
        print(df_sorted[['name', 'model_type', 'test_mae', 'val_mae', 'train_time']].to_string(index=False))
        print(f"{'='*70}\n")
        
        print(f"✅ Best model: {df_sorted.iloc[0]['name']} (Test MAE: {df_sorted.iloc[0]['test_mae']:.4f})")
    except Exception as e:
        print(f"Could not load results: {e}")


if __name__ == '__main__':
    main()