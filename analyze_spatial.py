import numpy as np
import matplotlib.pyplot as plt

# Load allosteric scores
data = np.load('test_output_3d/allosteric_scores.model_idx_0.npz')
scores = data['allosteric_scores']

# Basic statistics
min_score = scores.min()
max_score = scores.max()
mean_score = scores.mean()
std_score = scores.std()

print(f"Allosteric score statistics:")
print(f"  Min: {min_score:.4f}")
print(f"  Max: {max_score:.4f}")
print(f"  Mean: {mean_score:.4f}")
print(f"  Std Dev: {std_score:.4f}")

# Find top 10 allosteric sites
top_indices = np.argsort(scores)[-10:][::-1]  # Reverse to get highest first
top_scores = scores[top_indices]

print("\nTop 10 allosteric sites:")
for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
    print(f"#{i+1}: Residue {idx} with score {score:.4f}")

# Calculate proportion of residues with high scores
high_score_threshold = mean_score + std_score
high_score_count = np.sum(scores > high_score_threshold)
high_score_percentage = (high_score_count / len(scores)) * 100

print(f"\nProportion of high-scoring residues (>{high_score_threshold:.4f}): {high_score_percentage:.1f}%")
