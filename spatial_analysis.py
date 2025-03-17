import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os

# Create a function to parse CA coordinates from a CIF file (simplified)
def extract_ca_coords_from_cif(cif_path):
    coords = []
    residue_indices = []
    current_residue = None
    
    with open(cif_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) >= 11 and parts[3] == 'CA':
                    x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
                    residue_idx = int(parts[8])
                    
                    # Only add if it's a new residue
                    if residue_idx \!= current_residue:
                        coords.append([x, y, z])
                        residue_indices.append(residue_idx)
                        current_residue = residue_idx
    
    return np.array(coords), np.array(residue_indices)

# Load allosteric scores
allosteric_data = np.load('test_output_3d/allosteric_scores.model_idx_0.npz')
allosteric_scores = allosteric_data['allosteric_scores']

# Extract CA coordinates from the CIF file
ca_coords, residue_indices = extract_ca_coords_from_cif('test_output_3d/pred.model_idx_0.cif')

# Find top 10 allosteric sites
top_indices = np.argsort(allosteric_scores)[-10:]
top_scores = allosteric_scores[top_indices]

# Calculate pairwise distances between all CA atoms
dist_matrix = squareform(pdist(ca_coords))

# For each top site, find if it has 3D neighbors that are also high-scoring
print("3D spatial analysis of top allosteric sites:")
print("-" * 50)

# Convert residue_indices to a list for easier lookup
residue_list = residue_indices.tolist()

for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
    # Find the corresponding index in the CA coordinates array
    try:
        ca_idx = residue_list.index(idx) if idx in residue_list else None
    except ValueError:
        ca_idx = None
    
    if ca_idx is not None:
        # Find all residues within 10Å
        neighbors = np.where(dist_matrix[ca_idx] < 10.0)[0]
        
        # Get the scores of these neighbors
        neighbor_residue_indices = residue_indices[neighbors]
        valid_scores = []
        
        for n_idx in neighbor_residue_indices:
            if n_idx < len(allosteric_scores) and n_idx \!= idx:
                valid_scores.append((n_idx, allosteric_scores[n_idx]))
        
        # Sort neighbors by score
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Site #{i+1}: Residue {idx} (Score: {score:.4f})")
        print(f"  Has {len(valid_scores)} nearby residues within 10Å")
        
        if valid_scores:
            high_scoring_neighbors = [(n_idx, n_score) for n_idx, n_score in valid_scores 
                                     if n_score > allosteric_scores.mean() + allosteric_scores.std()]
            
            if high_scoring_neighbors:
                print(f"  High-scoring neighbors (>{allosteric_scores.mean() + allosteric_scores.std():.4f}):")
                for n_idx, n_score in high_scoring_neighbors[:5]:  # Show top 5
                    print(f"    Residue {n_idx}: {n_score:.4f}")
            else:
                print("  No high-scoring neighbors")
        
        print()
    else:
        print(f"Site #{i+1}: Residue {idx} (Score: {score:.4f}) - CA atom not found")
        print()

# Check if the top sites form a potential allosteric pocket
# (Multiple high-scoring residues clustered in 3D space)
def find_3d_clusters(coords, residue_indices, scores, distance_threshold=10.0, score_threshold=None):
    if score_threshold is None:
        score_threshold = scores.mean() + scores.std()
    
    # Find high-scoring residues
    high_scoring_indices = np.where(scores > score_threshold)[0]
    
    # Find which of these have CA atoms
    clusters = []
    processed = set()
    
    for idx in high_scoring_indices:
        if idx in processed:
            continue
            
        try:
            ca_idx = residue_list.index(idx) if idx in residue_list else None
        except ValueError:
            ca_idx = None
            
        if ca_idx is not None:
            # Find neighbors within threshold
            neighbors = np.where(dist_matrix[ca_idx] < distance_threshold)[0]
            neighbor_residue_indices = residue_indices[neighbors]
            
            # Filter to include only high-scoring neighbors
            cluster = [idx]
            for n_idx in neighbor_residue_indices:
                if n_idx < len(scores) and n_idx \!= idx and scores[n_idx] > score_threshold and n_idx not in processed:
                    cluster.append(n_idx)
                    processed.add(n_idx)
            
            if len(cluster) > 1:  # Only include clusters with at least 2 residues
                clusters.append(cluster)
                processed.add(idx)
                
    return clusters

# Find 3D clusters of high-scoring residues
clusters = find_3d_clusters(ca_coords, residue_indices, allosteric_scores)

print("\n3D Spatial Clusters of High-Scoring Residues:")
print("-" * 50)
if clusters:
    for i, cluster in enumerate(clusters):
        avg_score = np.mean([allosteric_scores[idx] for idx in cluster])
        print(f"Cluster #{i+1}: {len(cluster)} residues with average score {avg_score:.4f}")
        print(f"  Residues: {cluster}")
        print()
else:
    print("No significant 3D clusters found")

