"""
Allosteric prediction head for Chai-1.

This module implements a Graph Neural Network (GNN) based approach to predict
allosteric binding sites in proteins directly within the Chai-1 model.
It leverages the confidence metrics (pLDDT, PAE) that Chai-1 already computes
and combines them with a graph-based representation of protein structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class GATConvLayer(nn.Module):
    """
    Graph Attention Convolution layer implementation using PyTorch primitives.
    
    This allows us to avoid requiring torch_geometric as a dependency while
    implementing the core functionality needed for graph-based processing.
    Enhanced with support for edge features to capture 3D spatial information.
    """
    def __init__(self, in_features: int, out_features: int, edge_features: int = 4, 
                 heads: int = 4, dropout: float = 0.2):
        super(GATConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.heads = heads
        self.dropout = dropout
        
        # Linear transformation for node features
        self.linear = nn.Linear(in_features, heads * out_features)
        
        # Edge feature processing
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, heads)
        )
        
        # Attention mechanism
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Transform node features
        x = self.linear(x).view(-1, self.heads, self.out_features)  # [N, heads, out_features]
        
        # Extract source and target nodes from edge_index
        src, dst = edge_index
        
        # Compute attention weights from node features
        alpha_src = (x[src] * self.att_src).sum(dim=-1)  # [E, heads]
        alpha_dst = (x[dst] * self.att_dst).sum(dim=-1)  # [E, heads]
        alpha = alpha_src + alpha_dst
        
        # Add edge feature attention if available
        if edge_attr is not None:
            # Process edge features
            edge_attention = self.edge_proj(edge_attr)  # [E, heads]
            alpha = alpha + edge_attention
        
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Normalize attention weights using softmax (per source node)
        alpha = softmax_by_src(alpha, src, num_nodes=x.size(0))
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention weights to neighbor features
        out = torch.zeros_like(x)
        for i in range(self.heads):
            # For each attention head
            alpha_h = alpha[:, i].unsqueeze(-1)  # [E, 1]
            message = x[dst, i] * alpha_h  # [E, out_features]
            
            # Aggregate messages (sum)
            out[:, i].index_add_(0, src, message)
        
        # Average over heads
        return out.mean(dim=1)


def softmax_by_src(src_values: torch.Tensor, indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Compute softmax of values grouped by source indices.
    This is similar to scatter_softmax in PyG but implemented with PyTorch primitives.
    """
    # Initialize output with negative infinity
    output = torch.full((num_nodes, src_values.size(1)), -float('inf'), 
                        device=src_values.device)
    
    # Fill in values
    output[indices] = src_values
    
    # Apply softmax along dim=0 grouped by indices
    output = torch.softmax(output, dim=0)
    
    # Extract values for original edges
    return output[indices]


class AllostericHead(nn.Module):
    """
    Allosteric prediction head for Chai-1 model with enhanced 3D spatial features.
    
    This module implements a Graph Neural Network (GNN) based approach to predict
    allosteric binding sites in proteins. It leverages Chai-1's confidence metrics
    (pLDDT, PAE) and combines them with a 3D graph-based representation of the protein.
    """
    
    def __init__(self, 
                node_features: int = 25,  # Increased for 3D coordinates
                hidden_dim: int = 64,
                edge_features: int = 4,  # Distance + 3D direction vector
                num_layers: int = 3,
                dropout: float = 0.2):
        """
        Initialize the allosteric prediction head with support for 3D spatial features.
        
        Args:
            node_features: Number of input node features (includes 3D coordinates)
            hidden_dim: Hidden dimension for GNN layers
            edge_features: Number of edge features (distance + direction vector)
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(AllostericHead, self).__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Spatial embedding to process 3D coordinates specially
        self.spatial_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # GNN layers with residual connections
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConvLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                edge_features=edge_features,
                heads=4,
                dropout=dropout
            ))
        
        # Projection for pLDDT features
        self.plddt_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Projection for PAE features
        self.pae_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layer
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
               node_features: torch.Tensor,
               edge_index: torch.Tensor,
               plddt: torch.Tensor,
               edge_attr: Optional[torch.Tensor] = None,
               pae: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for allosteric site prediction with 3D spatial features.
        
        Args:
            node_features: Tensor of shape [num_nodes, node_features] 
                          (includes one-hot residue type, sequence position, 3D coordinates)
            edge_index: Tensor of shape [2, num_edges] containing source and target indices
            plddt: Tensor of shape [num_nodes] containing pLDDT scores
            edge_attr: Optional tensor of shape [num_edges, edge_features] with edge features
                      (distance and direction vectors)
            pae: Optional tensor of shape [num_nodes, num_nodes] containing PAE matrix
            
        Returns:
            Tensor of shape [num_nodes] with allosteric site scores (0-1)
        """
        # Safety check: if we have empty inputs, return outputs with proper shape
        if node_features.shape[0] == 0 or edge_index.shape[1] == 0:
            print("WARNING: Empty graph received in AllostericHead. Creating placeholder scores.")
            # Try to estimate the number of tokens from plddt
            if plddt is not None and plddt.numel() > 0:
                num_tokens = plddt.shape[0]
                print(f"Creating placeholder scores for {num_tokens} tokens")
                return torch.zeros(num_tokens, device=node_features.device)
            else:
                print("WARNING: Cannot determine token count. Returning minimal placeholder.")
                # Just return something with at least 1 dimension
                return torch.zeros(1, device=node_features.device)
            
        # Check that edge indices are within bounds
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= node_features.shape[0]:
                print(f"WARNING: Edge index {max_idx} out of bounds for {node_features.shape[0]} nodes. Clamping.")
                edge_index = torch.clamp(edge_index, max=node_features.shape[0]-1)
        
        # Ensure pLDDT is properly shaped
        if plddt.dim() == 1:
            plddt = plddt.unsqueeze(1)  # [num_nodes, 1]
            
        # Extract 3D coordinates from node features to process separately
        # The last 3 features of node_features are the 3D coordinates
        coords_3d = node_features[:, -3:]
        
        # Process the 3D coordinates with a dedicated network
        spatial_features = self.spatial_embedding(coords_3d)
        
        # Process the base node features
        base_node_features = self.node_embedding(node_features)
        
        # Add the spatial features with residual connection
        x = base_node_features + spatial_features
        
        # Apply GNN layers with residual connections
        for conv in self.convs:
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x + x_res)  # Residual connection
        
        # Process pLDDT scores
        plddt_features = self.plddt_projection(plddt)
        
        # Process PAE (if provided)
        if pae is not None:
            # For now, just use mean PAE per residue
            mean_pae = pae.mean(dim=1, keepdim=True)
            pae_features = self.pae_projection(mean_pae)
        else:
            # If PAE not provided, use zero features of the same shape
            pae_features = torch.zeros_like(plddt_features)
        
        # Combine all features
        combined = torch.cat([x, plddt_features, pae_features], dim=1)
        
        # Apply fusion layers
        fused = self.fusion(combined)
        
        # Final prediction
        scores = self.prediction(fused).squeeze(-1)
        
        return scores


def build_protein_graph(
    residue_positions: torch.Tensor,
    residue_types: torch.Tensor,
    distance_threshold: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a protein graph from residue positions with 3D spatial features.
    
    Args:
        residue_positions: Tensor of shape [num_residues, 3] containing CA atom positions
        residue_types: Tensor of shape [num_residues] containing residue type indices
        distance_threshold: Distance threshold for connecting residues
        
    Returns:
        Tuple of (node_features, edge_index, edge_features)
    """
    num_residues = residue_positions.shape[0]
    device = residue_positions.device
    
    # Safety check for empty inputs
    if num_residues == 0:
        print("WARNING: Empty residue positions in build_protein_graph. Returning empty graph.")
        empty_features = torch.zeros((0, 25), device=device)  # 21 residue types + 1 position + 3 coords
        empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_edge_features = torch.zeros((0, 4), device=device)  # distance + 3D direction vector
        return empty_features, empty_edges, empty_edge_features
    
    # Special case for single residue
    if num_residues == 1:
        print("WARNING: Only one residue in build_protein_graph. Creating singleton graph.")
        # One-hot encode the single residue type (ensure it's long type for one_hot)
        residue_types_long = residue_types.clone().long()  # Must be long/int for one_hot
        node_features = F.one_hot(residue_types_long, num_classes=21).float()  # 20 amino acids + 1 for unknown
        pos_encoding = torch.tensor([[0.5]], device=device)  # Middle position encoding
        
        # Include 3D coordinates directly in node features
        node_features = torch.cat([node_features.float(), pos_encoding, residue_positions], dim=1)
        
        # Single node has no edges
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_features = torch.zeros((0, 4), device=device)  # Empty edge features
        return node_features, edge_index, edge_features
    
    # Standard case with multiple residues
    try:
        # One-hot encode residue types (clamping for safety)
        residue_types_clamped = torch.clamp(residue_types, min=0, max=20).long()  # Must be long/int for one_hot
        node_features = F.one_hot(residue_types_clamped, num_classes=21)  # 20 amino acids + 1 for unknown
        
        # Add positional encoding to node features (relative position in sequence)
        pos_encoding = torch.arange(num_residues, device=device).float() / max(1, num_residues - 1)
        pos_encoding = pos_encoding.unsqueeze(1)
        
        # Combine residue types, positional encoding, and 3D coordinates 
        node_features = torch.cat([node_features.float(), pos_encoding, residue_positions], dim=1)
        
        # Calculate pairwise distances
        distances = torch.cdist(residue_positions, residue_positions)
        
        # Connect residues within threshold distance
        src, dst = torch.where(distances < distance_threshold)
        
        # Remove self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]
        
        # Create edge index tensor
        if src.numel() > 0:
            edge_index = torch.stack([src, dst], dim=0)
            
            # Create edge features: distance and 3D direction vectors between nodes
            edge_distances = distances[src, dst].unsqueeze(1)  # [num_edges, 1]
            
            # Calculate 3D direction vectors between connected residues
            source_pos = residue_positions[src]
            target_pos = residue_positions[dst]
            direction_vectors = target_pos - source_pos  # [num_edges, 3]
            
            # Normalize direction vectors
            direction_norms = torch.norm(direction_vectors, dim=1, keepdim=True).clamp(min=1e-8)
            normalized_directions = direction_vectors / direction_norms
            
            # Combine distance and direction as edge features
            edge_features = torch.cat([edge_distances, normalized_directions], dim=1)  # [num_edges, 4]
            
        else:
            print("WARNING: No edges found in protein graph. Creating minimal edge set.")
            # Create minimal chain-like edges to ensure graph connectivity
            src = torch.arange(num_residues - 1, device=device)
            dst = torch.arange(1, num_residues, device=device)
            edge_index = torch.stack([src, dst], dim=0)
            
            # Calculate minimal edge features
            edge_distances = torch.zeros(len(src), 1, device=device)
            direction_vectors = torch.zeros(len(src), 3, device=device)
            
            # Calculate actual distances and directions for these edges
            for i in range(len(src)):
                source_idx = src[i]
                target_idx = dst[i]
                source_pos = residue_positions[source_idx]
                target_pos = residue_positions[target_idx]
                
                # Calculate distance
                diff = target_pos - source_pos
                dist = torch.norm(diff)
                edge_distances[i, 0] = dist
                
                # Calculate normalized direction vector
                direction_vectors[i] = diff / dist.clamp(min=1e-8)
            
            # Combine as edge features
            edge_features = torch.cat([edge_distances, direction_vectors], dim=1)
        
        return node_features, edge_index, edge_features
        
    except Exception as e:
        print(f"ERROR in build_protein_graph: {e}")
        # Fallback to minimal valid graph
        min_features = torch.zeros((1, 25), device=device)  # 21 + 1 + 3
        min_features[0, 0] = 1.0  # First residue type
        min_features[0, 21] = 0.5  # Middle position
        if residue_positions.shape[0] > 0:
            min_features[0, 22:25] = residue_positions[0]  # Use first position if available
        min_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        min_edge_features = torch.zeros((0, 4), device=device)
        return min_features, min_edges, min_edge_features