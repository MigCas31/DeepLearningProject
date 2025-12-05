import os
import numpy as np
import anndata as ad
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
# The train_model function will be called via subprocess, as direct import is not feasible or intended.
import subprocess
import sys

def run_stability_single_k(data_path, output_dir, n_runs=10, k=3):
    print(f"Starting stability analysis for k={k} with {n_runs} runs.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results: nmi_scores = [nmi_run1_vs_run2, nmi_run1_vs_run3, ...]
    
    # Load data once to get true labels if needed
    try:
        adata = ad.read_h5ad(data_path)
        n_samples = adata.shape[0]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\nProcessing k={k}...")
    run_assignments = []
    
    for i in range(n_runs):
        run_dir = os.path.join(output_dir, f"k_{k}_run_{i}")
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"  Run {i+1}/{n_runs}")
        
        # Call train.py via subprocess to ensure clean state
        cmd = [
            "python3", "train.py",
            "--n_archetypes", str(k),
            "--save_path", run_dir,
            "--data_path", data_path,
            "--epochs", "5000" 
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Load S matrix
            s_path = os.path.join(run_dir, "S_matrix.npy")
            if os.path.exists(s_path):
                S = np.load(s_path)
                # S is (n_archetypes, n_samples), transpose to (n_samples, n_archetypes)
                W = S.T
                # Get hard assignments
                assignments = np.argmax(W, axis=1)
                run_assignments.append(assignments)
            else:
                print(f"    Warning: S_matrix.npy not found for run {i}")
        except subprocess.CalledProcessError as e:
            print(f"    Error in run {i}: {e}")
    
    # Calculate pairwise NMI between all runs for this k
    if len(run_assignments) < 2:
        print(f"  Not enough successful runs for k={k} to calculate stability.")
        return
        
    nmis = []
    for i in range(len(run_assignments)):
        for j in range(i + 1, len(run_assignments)):
            nmi = normalized_mutual_info_score(run_assignments[i], run_assignments[j])
            nmis.append(nmi)
    
    avg_nmi = np.mean(nmis)
    print(f"  Average Stability NMI for k={k}: {avg_nmi:.4f}")

    # Save results for this k
    results = {"k": k, "nmis": nmis, "avg_nmi": avg_nmi}
    save_path = os.path.join(output_dir, f"stability_results_k_{k}.npy")
    np.save(save_path, results)
    print(f"Saved results to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NMI Stability for Single K")
    parser.add_argument("--data_path", type=str, default="9ng_atac_like.h5ad", help="Path to data")
    parser.add_argument("--output_dir", type=str, default="./results_stability", help="Output directory")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--k", type=int, required=True, help="Number of archetypes to test")
    args = parser.parse_args()
    
    run_stability_single_k(args.data_path, args.output_dir, args.n_runs, args.k)
