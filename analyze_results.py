import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Load the data
allosteric_data = np.load('test_output_3d/allosteric_scores.model_idx_0.npz')
allosteric_scores = allosteric_data['allosteric_scores']

# Find the top 10 allosteric sites
top_indices = np.argsort(allosteric_scores)[-10:]
top_scores = allosteric_scores[top_indices]

print(f"Top 10 allosteric sites:")
for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
    print(f"#{i+1}: Residue {idx} with score {score:.4f}")

# Find sites that cluster together in 3D space
# For this analysis we'd need coordinates, but we can identify sequence-adjacent sites
clusters = []
current_cluster = [top_indices[0]]

for i in range(1, len(top_indices)):
    if abs(top_indices[i] - top_indices[i-1]) <= 4:  # Consider residues within 4 positions as a cluster
        current_cluster.append(top_indices[i])
    else:
        if len(current_cluster) > 1:  # Only count clusters with at least 2 residues
            clusters.append(current_cluster)
        current_cluster = [top_indices[i]]

# Add the last cluster if it exists
if len(current_cluster) > 1:
    clusters.append(current_cluster)

print(f"\nFound {len(clusters)} potential allosteric site clusters in sequence:")
for i, cluster in enumerate(clusters):
    avg_score = np.mean(allosteric_scores[cluster])
    print(f"Cluster #{i+1}: Residues {cluster} with average score {avg_score:.4f}")

# Analyze score distribution
print(f"\nAllosteric score distribution:")
print(f"  Min: {allosteric_scores.min():.4f}")
print(f"  Max: {allosteric_scores.max():.4f}")
print(f"  Mean: {allosteric_scores.mean():.4f}")
print(f"  Std Dev: {allosteric_scores.std():.4f}")

# Calculate score differential (how much the top sites stand out)
diff_from_mean = (allosteric_scores.max() - allosteric_scores.mean()) / allosteric_scores.std()
print(f"\nTop site stands out by {diff_from_mean:.2f} standard deviations from the mean")

