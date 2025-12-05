import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import anndata as ad
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

# Add the model directory to sys.path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Archetypal-Analysis-For-Binary-Data-main', 'Python'))

from src.methods.AABernoulli import Bernoulli_Archetypal_Analysis
from src.methods.AALS import AALS

def preprocess_text(text, ps, stop_words):
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered_tokens)

def load_and_process_data(data_path="9ng_atac_like.h5ad"):
    if os.path.exists(data_path):
        print(f"Loading processed data from {data_path}...")
        return ad.read_h5ad(data_path)
    

    # Fetch data
    newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    labels = newsgroups_train.target
    label_names = np.array(newsgroups_train.target_names)

    # Merge groups
    merge = {
        'comp.graphics': 'comp', 'comp.os.ms-windows.misc': 'comp', 'comp.sys.ibm.pc.hardware': 'comp', 'comp.sys.mac.hardware': 'comp',
        'comp.windows.x': 'comp', 'sci.space': 'sci', 'sci.crypt': 'sci', 'sci.electronics': 'sci',
        'sci.med': 'sci', 'rec.sport.baseball': 'rec', 'rec.sport.hockey': 'rec', 'rec.motorcycles': 'rec',
        'rec.autos': 'rec', 'talk.politics.mideast': 'talk.politics', 'talk.politics.misc': 'talk.politics',
        'talk.politics.guns': 'talk.politics', 'misc.forsale': 'misc', 'soc.religion.christian': 'religion',
        'alt.atheism': 'religion'
    }
    new_labels = np.array([merge[label_name] if label_name in merge else label_name for label_name in label_names[labels]])

    # Preprocessing
    print("Preprocessing text...")
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Use a simple list comprehension for speed if tqdm fails or for simplicity
    preprocessed_data = []
    for text in tqdm(newsgroups_train.data, desc="Tokenizing and Stemming"):
        preprocessed_data.append(preprocess_text(text, ps, stop_words))

    # Vectorization
    print("Vectorizing...")
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X_counts = vectorizer.fit_transform(preprocessed_data)

    # LDA
    print("Running LDA...")
    n_topics = 25
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="batch", random_state=0)
    X_lda = lda.fit_transform(X_counts)

    # Create AnnData
    obs_df = pd.DataFrame({"category": new_labels})
    var_df = pd.DataFrame(index=[f"topic_{i}" for i in range(n_topics)])
    
    adata = ad.AnnData(X_lda, obs=obs_df, var=var_df)
    
    print(f"Saving processed data to {data_path}...")
    adata.write(data_path)
    
    return adata

def main():
    parser = argparse.ArgumentParser(description="Train Archetypal Analysis on 20 Newsgroups data")
    parser.add_argument("--data_path", type=str, default="9ng_atac_like.h5ad", help="Path to .h5ad data file")
    parser.add_argument("--model", type=str, default="bernoulli", choices=["bernoulli", "aals"], help="Model type: bernoulli or aals")
    parser.add_argument("--n_archetypes", type=int, default=25, help="Number of archetypes")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum iterations")
    parser.add_argument("--save_path", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--test_mode", action="store_true", help="Run on a small subset for testing")
    
    args = parser.parse_args()

    # Load data
    adata = load_and_process_data(args.data_path)
    
    if args.test_mode:
        print("Running in TEST MODE with subset of data...")
        adata = adata[:100, :]

    X = adata.X
    
    # Ensure X is appropriate type/format
    if isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.double)
    else:
        X_tensor = torch.tensor(X.toarray(), dtype=torch.double)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_tensor = X_tensor.to(device)
    X_tensor = X_tensor.T
    
    print(f"Data shape for model: {X_tensor.shape}")

    os.makedirs(args.save_path, exist_ok=True)

    if args.model == "bernoulli":
        print(f"Training Bernoulli Archetypal Analysis with {args.n_archetypes} archetypes...")
        C, S, losses = Bernoulli_Archetypal_Analysis(
            X_tensor, 
            n_arc=args.n_archetypes, 
            maxIter=args.epochs
        )
    elif args.model == "aals":
        print(f"Training AALS with {args.n_archetypes} archetypes...")
        # AALS signature: AALS(X, n_arc, C=None, S=None, gridS=False, maxIter=1000, device='cpu')
        C, S, losses, explained_var = AALS(
            X_tensor,
            n_arc=args.n_archetypes,
            maxIter=args.epochs,
            device=device
        )

    # Save results
    print("Saving results...")
    
    if torch.is_tensor(C): C = C.cpu().numpy()
    if torch.is_tensor(S): S = S.cpu().numpy()
    
    np.save(os.path.join(args.save_path, "C_matrix.npy"), C)
    np.save(os.path.join(args.save_path, "S_matrix.npy"), S)
    np.save(os.path.join(args.save_path, "losses.npy"), np.array(losses))
    
    print("Training complete.")

if __name__ == "__main__":
    main()
