# Allosteric Binding Site Prediction in Chai-1

## Overview

This documentation describes the implementation of allosteric binding site prediction functionality directly within the Chai-1 model. This enhancement allows Chai-1 to predict not only protein structures but also regions that are likely to function as allosteric binding sites.

## What Are Allosteric Binding Sites?

Allosteric binding sites are regions on proteins that:
- Are often located far from the protein's active site
- When bound by ligands, can regulate protein function through long-range structural interactions
- May only appear in certain protein conformational states
- Are challenging to identify using traditional binding site prediction methods

## Implementation Details

### Technical Approach

Our implementation integrates a 3D-aware Graph Neural Network (GNN) with Chai-1's existing structure prediction capabilities. This approach:

1. Leverages Chai-1's already-computed confidence metrics (pLDDT, PAE)
2. Represents proteins as 3D spatial graphs where:
   - Nodes are residues with features including residue type, sequence position, and 3D coordinates
   - Edges connect residues within a specified distance and contain spatial features (distance and direction vectors)
3. Uses graph attention mechanisms that incorporate 3D spatial information to capture long-range interactions
4. Specially processes 3D coordinates with dedicated neural network layers
5. Combines structural, spatial, and graph information to predict per-residue probabilities of being part of an allosteric site

### Key Components

1. **Allosteric Head Module** (`model/allosteric_head.py`)
   - Custom 3D-aware Graph Attention Network implementation using PyTorch primitives
   - Processes protein graphs with dedicated spatial feature processing
   - Integrates 3D coordinates and direction vectors as first-class features
   - Projects and combines pLDDT and PAE features from Chai-1
   - Performs specialized processing of spatial features through dedicated neural networks
   - Final prediction layer producing per-residue scores (0-1)

2. **Integration with Chai-1** (`chai1.py`)
   - Added `predict_allosteric_sites` parameter to `run_inference` function
   - Modified `StructureCandidates` class to include allosteric scores
   - Added allosteric prediction step in `run_folding_on_context`
   - Updated concatenation logic to handle allosteric scores

3. **3D Spatial Graph Building** (`model/allosteric_head.py`)
   - `build_protein_graph` function creates spatial graph representation from protein structure
   - Nodes: Residues with features including type, sequence position, and 3D coordinates
   - Edges: Connections between residues within 10Å, containing both:
     - Distance information (scalar)
     - Direction vector information (3D normalized vector)
   - Special processing for spatial relationships that preserves geometric information

## Usage

To use the allosteric site prediction functionality, simply add the `predict_allosteric_sites=True` parameter when calling `run_inference`:

```python
from pathlib import Path
from chai_lab.chai1 import run_inference

structure_candidates = run_inference(
    fasta_file=Path("my_protein.fasta"),
    output_dir=Path("./output"),
    predict_allosteric_sites=True,  # Enable allosteric site prediction
)

# Access the predicted allosteric scores
allosteric_scores = structure_candidates.allosteric_scores  # Shape: [num_models, num_residues]
```

The allosteric scores are also saved to NPZ files for each model in the output directory:
- `allosteric_scores.model_idx_0.npz`
- `allosteric_scores.model_idx_1.npz`
- etc.

## Technical Design Decisions

1. **Enhanced 3D Spatial Representation**: Our implementation captures the 3D spatial relationships between residues by:
   - Including 3D coordinates directly in node features
   - Representing edge relationships with both distance and direction vectors
   - Processing spatial features through dedicated neural network layers

2. **Custom GNN Implementation**: We implemented 3D-aware graph attention convolution using PyTorch primitives rather than requiring torch_geometric as a dependency.

3. **Integration with Confidence Metrics**: We leverage Chai-1's pLDDT and PAE scores, which provide valuable information about structural confidence and residue-residue spatial relationships.

4. **Post-Diffusion Processing**: The allosteric prediction is performed after the diffusion process is complete, allowing it to benefit from the final predicted structure.

5. **Memory Efficiency**: Calculations are performed per-model to minimize memory usage, with careful handling of GPU memory for 3D spatial computations.

## Future Work

1. **Training Pipeline**: The current implementation needs to be trained on a dataset of known allosteric sites.

2. **Additional Features**: Integration of evolutionary conservation data and chemical properties.

3. **Visualization**: Tools to visualize predicted allosteric sites alongside protein structures.

4. **Integration with Pocket Detection**: Combining with dedicated pocket detection algorithms for improved performance.

## Troubleshooting

If you encounter errors related to atom name access when running allosteric prediction, ensure:

1. You're accessing atom names through `feature_context.structure_context.atom_ref_name` (not directly through `feature_context.atom_name`).

2. You're getting residue types from `feature_context.structure_context.token_residue_type` (not through feature generators).

3. Graph building and residue identification code works with the specific data structures used by Chai-1's atom and residue representations.