import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_classification(data_path, results_dir):
    print(f"Loading data from {data_path}...")
    try:
        adata = ad.read_h5ad(data_path)
        true_labels = adata.obs['category'].values
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loading results from {results_dir}...")
    try:
        # Load S matrix (n_archetypes, n_samples) - Reconstruction weights
        S = np.load(os.path.join(results_dir, "S_matrix.npy"))
        # Transpose to (n_samples, n_archetypes) for easier handling
        W = S.T
    except FileNotFoundError:
        print("S_matrix.npy not found.")
        return

    # Check shapes
    if len(true_labels) != W.shape[0]:
        # If running in test mode locally, W might be smaller than the full dataset
        print(f"Warning: Number of labels ({len(true_labels)}) does not match S matrix ({W.shape[0]}).")
        print("Assuming results correspond to the first N samples (test mode).")
        true_labels = true_labels[:W.shape[0]]
        X = adata.X[:W.shape[0]]
    else:
        X = adata.X

    # Calculate Archetype Feature Profiles (Archetypes = X.T @ C)
    # We need C for this.
    try:
        C = np.load(os.path.join(results_dir, "C_matrix.npy"))
        # Ensure C matches X dimensions if we truncated X
        if C.shape[0] != X.shape[0]:
             C = C[:X.shape[0], :]
    except FileNotFoundError:
        print("C_matrix.npy not found (needed for topic analysis).")
        C = None

    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Archetype_features = X.T @ C
    if C is not None:
        archetype_features = X.T @ C
    else:
        archetype_features = None
        
    feature_names = adata.var_names

    # 1. Assign each document to the archetype with the highest weight
    # Use W (S.T) for classification
    predicted_archetypes = np.argmax(W, axis=1)
    n_archetypes = W.shape[1]

    # 2. Map each archetype to the most frequent true label (Majority Vote)
    archetype_to_label = {}
    print("\nArchetype -> Label Mapping:")
    print("-" * 30)
    
    for i in range(n_archetypes):
        # Get indices of documents assigned to this archetype
        indices = np.where(predicted_archetypes == i)[0]
        
        if len(indices) == 0:
            print(f"Archetype {i}: No documents assigned.")
            archetype_to_label[i] = "Unassigned"
            continue
            
        # Get true labels for these documents
        labels_in_cluster = true_labels[indices]
        
        # Find label distribution
        counts = pd.Series(labels_in_cluster).value_counts()
        most_common = counts.index[0]
        purity = counts.iloc[0] / len(indices)
        
        archetype_to_label[i] = most_common
        print(f"Archetype {i}: {most_common} (Purity: {purity:.2%}, Count: {len(indices)})")
        print(f"    Top 3 Labels: {counts.head(3).to_dict()}")
        
        # Get top topics for this archetype
        if archetype_features is not None:
            top_topic_indices = np.argsort(archetype_features[:, i])[::-1][:3]
            top_topics = [f"{feature_names[idx]} ({archetype_features[idx, i]:.2f})" for idx in top_topic_indices]
            print(f"    Top 3 Topics: {', '.join(top_topics)}")

    # 3. Predict class based on mapping
    predicted_labels = np.array([archetype_to_label[a] for a in predicted_archetypes])

    # 4. Calculate Accuracy
    # Filter out 'Unassigned' if any (though argmax always assigns something)
    valid_mask = predicted_labels != "Unassigned"
    acc = accuracy_score(true_labels[valid_mask], predicted_labels[valid_mask])
    
    print("\n" + "=" * 30)
    print(f"Classification Accuracy: {acc:.2%}")
    print("=" * 30)

    # 5. Confusion Matrix
    labels = np.unique(true_labels)
    cm = confusion_matrix(true_labels[valid_mask], predicted_labels[valid_mask], labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
    plt.xlabel('Predicted Label (via Archetype)')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "classification_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"\nSaved confusion matrix to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Archetypal Analysis Classification Performance")
    parser.add_argument("--data_path", type=str, default="9ng_atac_like.h5ad", help="Path to .h5ad data file")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory containing results")
    args = parser.parse_args()
    
    evaluate_classification(args.data_path, args.results_dir)
