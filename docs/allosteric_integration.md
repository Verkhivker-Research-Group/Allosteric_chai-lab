# Allosteric Site Prediction Integration in Chai-1

This document explains the integration of allosteric binding site prediction functionality into the Chai-1 protein structure prediction system.

## Overview

The allosteric site prediction uses a Graph Neural Network (GNN) to identify regions in proteins likely to function as allosteric binding sites. This feature operates on the final predicted structures from Chai-1, leveraging the confidence metrics (pLDDT and PAE) that are already computed.

## Implementation Details

The integration adds the following components:

1. **AllostericHead model** (`chai_lab/model/allosteric_head.py`)
   - A specialized GNN for predicting allosteric sites
   - Custom graph attention convolution layers
   - Functions for building protein graphs from structure

2. **Integration with main prediction pipeline** (`chai_lab/chai1.py`)
   - Added `predict_allosteric_sites` parameter to `run_inference`
   - Modified `StructureCandidates` to include allosteric scores
   - Added allosteric prediction step in `run_folding_on_context`

3. **Documentation** (`docs/allosteric_prediction.md`)
   - Detailed explanation of the allosteric prediction approach
   - Usage instructions

## Changes Made During Integration

The integration process required several adjustments to handle the unique data structures in Chai-1:

1. **CA Atom Identification**:
   - Modified atom extraction to properly identify Cα atoms for each residue
   - Implemented fallback mechanisms when CA atoms are not explicitly marked

2. **Graph Construction**:
   - Added tensor type conversions for one-hot encoding
   - Improved mapping between atom indices and residue indices
   - Added handling for single-residue and empty graph cases

3. **GPU Optimization**:
   - Ensured consistent device placement for optimal performance on GPU
   - Reduced data transfers between CPU and GPU

4. **Error Handling and Graceful Fallbacks**:
   - Added checks for tensor shapes and data types
   - Implemented fallbacks for edge cases
   - Ensured proper tensor dimensions in all cases

## How to Use

To use the allosteric site prediction, simply add the `predict_allosteric_sites=True` parameter to your `run_inference` call:

```python
from pathlib import Path
from chai_lab.chai1 import run_inference

structure_candidates = run_inference(
    fasta_file=Path("my_protein.fasta"),
    output_dir=Path("./output"),
    predict_allosteric_sites=True,  # Enable allosteric prediction
)

# Access the predicted allosteric scores
allosteric_scores = structure_candidates.allosteric_scores  # Shape: [num_models, num_residues]
```

### Output Files

The allosteric prediction generates additional output files:

1. **NPZ files with allosteric scores** for each model:
   - `output/allosteric_scores.model_idx_0.npz`
   - `output/allosteric_scores.model_idx_1.npz`
   - etc.

2. **Visualization** can be generated with standard matplotlib code:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load allosteric scores
   data = np.load('output/allosteric_scores.model_idx_0.npz')
   scores = data['allosteric_scores']
   
   # Create visualization
   plt.figure(figsize=(10, 5))
   plt.bar(range(len(scores)), scores)
   plt.xlabel('Residue Index')
   plt.ylabel('Allosteric Site Score')
   plt.title('Predicted Allosteric Binding Sites')
   plt.savefig('output/allosteric_scores_plot.png')
   ```

## Testing

You can test the allosteric prediction functionality using the included test script:

```bash
python3 test_allosteric.py
```

This script:
1. Runs Chai-1 on a simple test protein
2. Enables allosteric site prediction
3. Generates a visualization of the predicted scores
4. Prints statistics about the predictions

## Limitations and Future Work

1. **Training Data**: The current implementation needs to be trained on a dataset of known allosteric sites.

2. **Performance**: The GNN adds some computational overhead to the prediction process.

3. **Future Improvements**:
   - Integration with MSA information for evolutionary context
   - Enhanced visualization tools
   - Integration with pocket detection algorithms

## Troubleshooting

If you encounter issues with the allosteric prediction:

1. **Memory Issues**: For large proteins, you may need to reduce the batch size or use `low_memory=True`.

2. **Device Errors**: Ensure tensors are consistently on the same device (CPU or GPU).

3. **CA Atom Detection**: The code attempts to identify CA atoms either by name or by position. If your structure uses non-standard atom naming, you might need to adjust the detection logic.

4. **Type Errors**: Ensure that residue type tensors are properly converted to integer types for one-hot encoding.