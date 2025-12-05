import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

def plot_sweep_results(results_dir="./results"):
    print(f"Scanning {results_dir} for sweep results...")
    
    # Find all k_* directories
    dirs = glob.glob(os.path.join(results_dir, "k_*"))
    
    data = []
    
    for d in dirs:
        # Extract k from directory name
        dirname = os.path.basename(d)
        match = re.match(r"k_(\d+)", dirname)
        if not match:
            continue
            
        k = int(match.group(1))
        
        # Load losses
        loss_path = os.path.join(d, "losses.npy")
        if not os.path.exists(loss_path):
            print(f"Skipping {dirname}: losses.npy not found")
            continue
            
        try:
            losses = np.load(loss_path)
            if len(losses) == 0:
                print(f"Skipping {dirname}: Empty losses file")
                continue
                
            final_loss = losses[-1]
            data.append({"k": k, "loss": final_loss})
            print(f"Found k={k}, Final Loss={final_loss:.4f}")
        except Exception as e:
            print(f"Error loading {dirname}: {e}")
            
    if not data:
        print("No valid results found.")
        return

    # Sort by k
    data.sort(key=lambda x: x["k"])
    
    ks = [d["k"] for d in data]
    losses = [d["loss"] for d in data]
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(ks, losses, 'o-', linewidth=2, markersize=8)
    
    plt.title("Archetypal Analysis: Loss vs. Number of Archetypes (Elbow Method)", fontsize=14)
    plt.xlabel("Number of Archetypes (k)", fontsize=12)
    plt.ylabel("Final Loss", fontsize=12)
    plt.xticks(ks)  # Ensure all k values are shown on x-axis
    
    # Annotate points
    for k, loss in zip(ks, losses):
        plt.annotate(f"{loss:.0f}", (k, loss), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "sweep_loss_plot.png")
    plt.savefig(save_path)
    print(f"\nSaved sweep plot to {save_path}")

if __name__ == "__main__":
    plot_sweep_results()
