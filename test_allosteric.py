#!/usr/bin/env python3
"""
Simple test script for allosteric binding site prediction with Chai-1.
"""

import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from chai_lab.chai1 import run_inference

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_allosteric_prediction():
    """Test the allosteric prediction functionality."""
    
    # Use a simple protein sequence from examples
    fasta_file = Path("examples/covalent_bonds/1CKJ_A.fasta")
    output_dir = Path("test_output_3d")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Force GPU 7 usage
    device = "cuda:7" if torch.cuda.is_available() and torch.cuda.device_count() > 7 else "cpu"
    print(f"Using device: {device}")

    # Run inference with allosteric prediction
    print("Running Chai-1 inference with allosteric prediction...")
    try:
        candidates = run_inference(
            fasta_file=fasta_file,
            output_dir=output_dir,
            use_esm_embeddings=False,  # Set to False to avoid ESM loading issues
            num_diffn_samples=2,  # Use fewer samples for test
            predict_allosteric_sites=True,
            seed=42,
            device=device  # Explicitly set to GPU 7
        )
        
        # Check if allosteric scores were generated
        if candidates.allosteric_scores is not None:
            print(f"Allosteric scores shape: {candidates.allosteric_scores.shape}")
            print(f"Mean allosteric score: {candidates.allosteric_scores[0].mean().item()}")
            print(f"Max allosteric score: {candidates.allosteric_scores[0].max().item()}")
            
            # Find the residues with highest allosteric scores
            top_k = 5
            top_indices = candidates.allosteric_scores[0].argsort(descending=True)[:top_k]
            print(f"Top {top_k} residues with highest allosteric scores:")
            for i, idx in enumerate(top_indices):
                print(f"  Residue {idx.item()}: {candidates.allosteric_scores[0][idx].item():.4f}")
            
            # Enhanced visualization for the 3D allosteric model
            top_model_idx = 0  # Best scoring model based on ranking
            scores = candidates.allosteric_scores[top_model_idx].cpu().numpy()
            residue_indices = np.arange(len(scores))
            
            # Basic scores plot
            plt.figure(figsize=(10, 6))
            plt.bar(residue_indices, scores)
            plt.xlabel("Residue Index")
            plt.ylabel("Allosteric Site Score")
            plt.title("Predicted Allosteric Binding Sites (3D Enhanced Model)")
            plt.savefig(output_dir / "allosteric_scores_plot.png")
            plt.close()
            
            # Calculate moving average for smoothed visualization
            window_size = 5
            if len(scores) >= window_size:
                weights = np.ones(window_size) / window_size
                smoothed_scores = np.convolve(scores, weights, mode='valid')
                smoothed_indices = residue_indices[window_size-1:]
                
                plt.figure(figsize=(10, 6))
                plt.plot(residue_indices, scores, 'b-', alpha=0.5, label='Raw scores')
                plt.plot(smoothed_indices, smoothed_scores, 'r-', linewidth=2, label='Moving average')
                plt.xlabel("Residue Index")
                plt.ylabel("Allosteric Site Score")
                plt.title("Allosteric Scores with Moving Average")
                plt.legend()
                plt.savefig(output_dir / "scores_with_moving_average.png")
                plt.close()
            
            # Distribution of scores
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=20, alpha=0.7, color='blue')
            plt.xlabel("Allosteric Score")
            plt.ylabel("Frequency")
            plt.title("Distribution of Allosteric Scores")
            plt.axvline(np.mean(scores), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
            plt.legend()
            plt.savefig(output_dir / "score_distribution.png")
            plt.close()
            
            # Identify peaks (potential allosteric sites)
            from scipy.signal import find_peaks
            if len(scores) > 3:  # Need at least a few points for peak finding
                # Find peaks with prominence to filter out noise
                peaks, properties = find_peaks(scores, height=np.mean(scores), prominence=0.05)
                
                if len(peaks) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(residue_indices, scores)
                    plt.plot(peaks, scores[peaks], "x", color='red', markersize=10, label=f'Peaks ({len(peaks)})')
                    
                    # Highlight regions around peaks
                    for peak in peaks:
                        plt.axvspan(max(0, peak-2), min(len(scores)-1, peak+2), color='red', alpha=0.2)
                    
                    plt.xlabel("Residue Index")
                    plt.ylabel("Allosteric Score")
                    plt.title("Identified Potential Allosteric Sites")
                    plt.legend()
                    plt.savefig(output_dir / "peaks_analysis.png")
                    plt.close()
                    
                    print(f"\nIdentified {len(peaks)} potential allosteric sites at residues:")
                    for i, peak in enumerate(peaks):
                        score = scores[peak]
                        print(f"  Site {i+1}: Residue {peak} (Score: {score:.4f})")
            
            # Create a detailed analysis figure comparing allosteric scores with pLDDT
            plt.figure(figsize=(12, 8))
            
            # Plot allosteric scores
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(residue_indices, scores, 'b-', linewidth=2)
            ax1.set_ylabel("Allosteric Score", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title("Allosteric Scores vs. Structure Confidence (pLDDT)")
            
            # Highlight top 3 allosteric sites
            if len(scores) > 0:
                top_3_indices = np.argsort(scores)[-3:]
                for idx in top_3_indices:
                    ax1.axvline(x=idx, color='blue', linestyle='--', alpha=0.5)
                    ax1.text(idx, scores[idx] + 0.05, f"{idx}", color='blue', 
                             horizontalalignment='center', fontsize=9)
            
            # Plot pLDDT on shared x-axis but separate y-axis
            ax2 = ax1.twinx()
            plddt_scores = candidates.plddt[top_model_idx].cpu().numpy()
            ax2.plot(residue_indices, plddt_scores, 'r-', linewidth=2)
            ax2.set_ylabel("pLDDT Score", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Plot correlation between allosteric scores and pLDDT
            ax3 = plt.subplot(2, 1, 2)
            ax3.scatter(plddt_scores, scores, alpha=0.7)
            ax3.set_xlabel("pLDDT Score")
            ax3.set_ylabel("Allosteric Score")
            ax3.set_title("Correlation: Structure Confidence vs. Allosteric Prediction")
            
            # Add correlation coefficient
            correlation = np.corrcoef(plddt_scores, scores)[0, 1]
            ax3.annotate(f"Correlation: {correlation:.3f}", 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_dir / "allosteric_detailed_plot.png")
            plt.close()
            
            print(f"Allosteric scores plot saved to {output_dir / 'allosteric_scores_plot.png'}")
            print(f"Detailed analysis saved to {output_dir / 'allosteric_detailed_plot.png'}")
            print("Test completed successfully!")
            return True
        else:
            print("ERROR: No allosteric scores were generated.")
            return False
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_allosteric_prediction()
