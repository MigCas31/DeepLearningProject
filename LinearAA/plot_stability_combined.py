import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from sklearn.metrics import normalized_mutual_info_score

def plot_combined_stability(results_dir="./results_stability"):
    abs_results_dir = os.path.abspath(results_dir)
    print(f"Scanning {abs_results_dir} for stability results...")
    
    # Find all run directories: k_{k}_run_{i}
    # We'll use a regex to parse folder names
    run_dirs = glob.glob(os.path.join(abs_results_dir, "k_*_run_*"))
    
    print(f"Found {len(run_dirs)} run directories.")
    
    # Organize by k
    # runs_by_k = { k: [path_to_run_0, path_to_run_1, ...] }
    runs_by_k = {}
    
    pattern = re.compile(r"k_(\d+)_run_(\d+)")
    
    for d in run_dirs:
        dirname = os.path.basename(d)
        match = pattern.match(dirname)
        if match:
            k = int(match.group(1))
            if k not in runs_by_k:
                runs_by_k[k] = []
            runs_by_k[k].append(d)
            
    if not runs_by_k:
        print("No run directories found.")
        return

    stability_scores = {}
    
    sorted_ks = sorted(runs_by_k.keys())
    
    for k in sorted_ks:
        print(f"Processing k={k}...")
        run_paths = runs_by_k[k]
        run_assignments = []
        
        for run_path in run_paths:
            s_path = os.path.join(run_path, "S_matrix.npy")
            if os.path.exists(s_path):
                try:
                    S = np.load(s_path)
                    # S is (n_archetypes, n_samples), transpose to (n_samples, n_archetypes) for argmax
                    # But wait, let's check shape. Usually S is (n_archetypes, n_samples).
                    # If we want assignments for each sample, we want argmax over archetypes.
                    # If S is (k, N), we want argmax(S, axis=0) -> shape (N,)
                    
                    # Let's verify shape assumption from stability_sweep.py:
                    # W = S.T  (N, k)
                    # assignments = np.argmax(W, axis=1)
                    # This implies S is (k, N).
                    
                    assignments = np.argmax(S, axis=0)
                    run_assignments.append(assignments)
                except Exception as e:
                    print(f"  Error loading {s_path}: {e}")
            else:
                # print(f"  Warning: {s_path} not found.")
                pass
        
        if len(run_assignments) < 2:
            print(f"  Not enough successful runs for k={k} (found {len(run_assignments)}). Skipping.")
            continue
            
        # Calculate pairwise NMI
        nmis = []
        for i in range(len(run_assignments)):
            for j in range(i + 1, len(run_assignments)):
                nmi = normalized_mutual_info_score(run_assignments[i], run_assignments[j])
                nmis.append(nmi)
        
        stability_scores[k] = nmis
        print(f"  Computed {len(nmis)} pairwise NMI scores. Mean: {np.mean(nmis):.4f}")

    if not stability_scores:
        print("No valid stability scores computed.")
        return

    # Sort by k
    sorted_keys = sorted(stability_scores.keys())
    
    # Prepare data for boxplot
    data = [stability_scores[k] for k in sorted_keys]
    labels = sorted_keys
    
    # Calculate means and 95% confidence intervals
    means = []
    cis = []
    for scores in data:
        if len(scores) > 1:
            mean = np.mean(scores)
            # Standard error of the mean
            sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
            # 95% CI roughly 1.96 * SEM (assuming normal distribution of means)
            ci = 1.96 * sem
            means.append(mean)
            cis.append(ci)
        else:
            means.append(np.mean(scores) if scores else 0)
            cis.append(0)

    # Create plot with both boxplot and error bars
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Boxplot for distribution
    # Use positions=k_values to ensure linear spacing
    k_values = sorted_keys
    bp = ax.boxplot(data, positions=k_values, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', color='blue', alpha=0.5),
                     medianprops=dict(color='darkblue'),
                     widths=0.6)
    
    # Overlay mean with error bars
    ax.errorbar(k_values, means, yerr=cis, fmt='o', color='red', capsize=5, label='Mean ± 95% CI', zorder=10)
    
    ax.set_title("Model Stability (Pairwise NMI) vs. Number of Archetypes", fontsize=16)
    ax.set_xlabel("Number of Archetypes (k)", fontsize=14)
    ax.set_ylabel("Pairwise NMI (Stability)", fontsize=14)
    ax.set_ylim(0, 1.05)
    
    # Set x-ticks to be the k values
    # User requested a range starting at 3
    if k_values:
        max_k = max(k_values)
        all_ks = np.arange(3, max_k + 1)
        ax.set_xticks(all_ks)
        ax.set_xticklabels(all_ks)
        ax.set_xlim(2.5, max_k + 0.5)
    
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    save_path = os.path.join(results_dir, "stability_plot_combined.png")
    fig.savefig(save_path, dpi=300)
    print(f"\nSaved combined stability plot to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Combined Stability Results")
    parser.add_argument("--results_dir", type=str, default="./results_stability", help="Directory containing results")
    args = parser.parse_args()
    
    plot_combined_stability(args.results_dir)
