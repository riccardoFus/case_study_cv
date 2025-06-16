"""
A set of functions for graph construction and manipulation using patch embedding.
This version includes a new implementation using a Vision Transformer for feature extraction.
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# Mock implementations for standalone execution
class MockConfig:
    def get(self, key):
        return 'graphs'
def get_config(): return MockConfig()
def get_device(): return 'cuda' if torch.cuda.is_available() else 'cpu'
def get_transformations(): return (lambda x: x), None, None, None, None


__all__ = ['image_to_graph_vit', 'extract_patches', 'get_node_edges']

"""
An implementation of a 3D Vision Transformer Autoencoder, designed to be a 
transformer-based counterpart to the convolutional AutoEncoder3D.
"""
from collections.abc import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import ViT
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Norm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityRanged
)

from sklearn.decomposition import PCA 

__all__ = ["ViTAutoEncoder3D"]


class ViTAutoEncoder3D(nn.Module):
    """
    A 3D Vision Transformer (ViT) based Autoencoder.

    This network is composed of a ViT-based encoder and a convolutional decoder.
    It's designed as a transformer-based alternative to a standard convolutional autoencoder,
    maintaining a similar configurable structure. The ViT encoder creates a latent
    representation of the input, and the convolutional decoder reconstructs the image
    from this latent space.

    The forward pass returns both the latent space and the reconstructed image,
    making it suitable for self-supervised pre-training.

    Args:
        spatial_dims: Number of spatial dimensions (must be 3).
        in_channels: Number of input channels.
        out_channels: Number of output channels for the reconstructed image.
        
        -- ViT Encoder Parameters --
        img_size: The size of the input image spatial dimensions (D, H, W).
                  Your data loader must provide patches of this exact size.
        patch_size: The size of the patches to be extracted from the input image.
        hidden_size: Dimension of the ViT latent space.
        mlp_dim: Dimension of the MLP layer in the ViT transformer block.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        
        -- Convolutional Decoder Parameters --
        decoder_channels: Sequence of channels for the decoder's convolutional blocks.
        decoder_strides: Sequence of strides for the transposed convolutions in the decoder.
                         Should have the same length as `decoder_channels`.
        
        -- General Parameters --
        act: Activation type and arguments. Defaults to PReLU.
        norm: Feature normalization type and arguments. Defaults to instance norm.
        dropout: Dropout ratio. Defaults to no dropout.
        bias: Whether to have a bias term in decoder convolution blocks. Defaults to True.
        name: A name for the model.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        # ViT Encoder parameters
        img_size: Sequence[int],
        patch_size: Sequence[int],
        hidden_size: int = 256,
        mlp_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 12,
        # Decoder parameters
        decoder_channels: Sequence[int] = (256, 128, 64),
        decoder_strides: Sequence[int] = (2, 2, 2),
        # General NN parameters
        act: tuple | str | None = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        name: str = 'ViTAutoEncoder3D',
    ) -> None:
        super().__init__()

        if spatial_dims != 3:
            raise ValueError("This implementation is designed for 3D spatial dimensions.")

        if len(decoder_channels) != len(decoder_strides):
            raise ValueError("Decoder channels and strides must have the same length.")

        self.name = name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.img_size = img_size
        

        # 1. --- Define the ViT Encoder ---
        self.encoder = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=False,  
            dropout_rate=dropout,
            spatial_dims=spatial_dims,
        )

        # 2. --- Define the Convolutional Decoder ---
        
        # Calculate the spatial size of the feature map after the ViT encoder
        self.patch_grid_size = [i // p for i, p in zip(img_size, patch_size)]
        
        self.decoder = self._get_decode_module(
            in_channels=hidden_size,
            channels=decoder_channels,
            strides=decoder_strides,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias
        )
        
        # Final layer to map to the desired number of output channels
        self.final_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=1, # 1x1x1 convolution
            conv_only=True,
            bias=True # Final layer usually has a bias
        )


    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int], **kwargs
    ) -> nn.Sequential:
        """
        Builds the decoder from a sequence of transposed convolutional blocks.
        """
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=layer_channels,
                out_channels=c,
                strides=s,
                kernel_size=s,      
                is_transposed=True,
                padding=0,          
                output_padding=0,   
                **kwargs
            )
            decode.add_module(f"decode_{i}", layer)
            layer_channels = c
            
        return decode


    def forward(self, x: torch.Tensor):
        """
        Forward pass for the ViT Autoencoder.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C_in, D, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - latent (torch.Tensor): The latent feature map from the encoder,
                  reshaped to its spatial grid `(B, hidden_size, D', H', W')`.
                - reconstruction (torch.Tensor): The reconstructed output image
                  with shape `(B, C_out, D, H, W)`.
        """
        # --- Encode ---
        # Encoder output is a sequence of tokens: (batch_size, num_patches, hidden_size)
        # Resize input if needed
        latent_tokens, _ = self.encoder(x)
        
        # --- Reshape latent space for decoder ---
        # Reshape token sequence into a 3D feature map
        # (B, num_patches, C) -> (B, C, num_patches) -> (B, C, D', H', W')
        b, n, c = latent_tokens.shape
        if c != self.hidden_size:
            raise ValueError("Latent token dimension mismatch.")
            
        latent_spatial = latent_tokens.transpose(1, 2).view(
            b, c, *self.patch_grid_size
        )
        
        # --- Decode ---
        reconstruction = self.decoder(latent_spatial)
        reconstruction = self.final_conv(reconstruction)
        
        return latent_spatial, reconstruction


def image_to_graph_vit(
    data: dict,
    vit_model: torch.nn.Module,
    patch_size=(8, 8, 8),
    k=10,
    num_important_features=20, 
    write_to_file: str = None,
    saved_path=None,
    saved_model=None
):
    """
    Constructs a graph by reducing the feature dimensionality of each node.

    This function feeds the entire image to a ViT, unpacks the feature grid for all
    nodes, and then uses PCA to reduce the feature dimension (e.g., from 256 to 20)
    before building the graph.

    Args:
        data (dict): Dictionary containing 'image' and 'label' paths.
        vit_model (torch.nn.Module): The pre-trained ViT model.
        patch_size (tuple): The size of patches corresponding to each node in the ViT grid.
        k (int): Number of nearest neighbors for graph edge construction.
        num_important_features (int): The target number of features after dimensionality reduction.
        write_to_file (str, optional): Path to save the graph and map files.
        saved_path (str, optional): Path to the directory with saved model weights.
        saved_model (str, optional): Specific model file name to load.

    Returns:
        tuple: A tuple containing:
            - graph (torch_geometric.data.Data or None): The constructed graph object.
            - centroids (numpy.ndarray or None): The 3D coordinates of the graph nodes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_pipeline = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image"], spatial_size=(112, 112, 72), mode='trilinear', align_corners=False),
        Resized(keys=["label"], spatial_size=(112, 112, 72), mode='nearest'),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
    ])

    try:
        processed_data = transform_pipeline(data)
    except Exception as e:
        print(f"Error processing data for subject {data.get('subject', 'N/A')}: {e}")
        return None, None

    image_volume = processed_data['image']
    label_volume = processed_data['label']

    image_tensor = torch.tensor(image_volume, dtype=torch.float).unsqueeze(0).to(device)

    if saved_path:
        model_files = [n for n in os.listdir(saved_path) if vit_model.name.upper() in n and n.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint found for {vit_model.name} in {saved_path}")
        last_model = saved_model if saved_model else sorted(model_files)[-1]
        vit_model.load_state_dict(torch.load(os.path.join(saved_path, last_model), map_location=device))

    vit_model.to(device)
    vit_model.eval()

    with torch.no_grad():
        output = vit_model(image_tensor)

    if isinstance(output, tuple):
        all_node_features = output[0]
    else:
        all_node_features = output

    if all_node_features.dim() == 5:
        b, d, d1, d2, d3 = all_node_features.shape
        all_node_features = all_node_features.permute(0, 2, 3, 4, 1).reshape(-1, d)
    elif all_node_features.dim() == 3:
        all_node_features = all_node_features.squeeze(0)
    else:
        raise ValueError(f"Unsupported shape for node features: {all_node_features.shape}")

    
    
    # Ensure the number of features to keep is not more than the original number
    original_feature_dim = all_node_features.shape[1]
    if num_important_features > original_feature_dim:
        print(f"Warning: Requested {num_important_features} features, but only {original_feature_dim} are available. Using {original_feature_dim}.")
        num_important_features = original_feature_dim

    # PCA works with NumPy on the CPU
    features_np = all_node_features.cpu().numpy()

    # Initialize and apply PCA
    pca = PCA(n_components=num_important_features)
    reduced_features_np = pca.fit_transform(features_np)
    
    # Convert back to a torch tensor on the correct device
    node_features = torch.tensor(reduced_features_np, dtype=torch.float, device=device)
    


    # --- Generate spatial and label info for ALL nodes ---
    grid_size = tuple(s // p for s, p in zip((112, 112, 72), patch_size))
    centroids, labels, voxels = [], [], []
    for z in range(grid_size[0]):
        for y in range(grid_size[1]):
            for x in range(grid_size[2]):
                coords = (z * patch_size[0], y * patch_size[1], x * patch_size[2])
                centroids.append(get_patch_centroid(coords, patch_size))
                labels.append(get_patch_labels(coords, patch_size, label_volume))
                voxels.append(get_patch_voxel_indices(coords, patch_size))

    centroids = np.array(centroids)

    # --- Build the final graph using ALL nodes but with REDUCED features ---
    graph = Data(
        x=node_features, # Use the new feature-reduced tensor
        edge_index=torch.tensor(get_node_edges(centroids=centroids, k=k).T, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long)
    )

    if write_to_file:
        fold = get_config().get('GRAPH_FOLDER')
        subject = data['subject']
        subject_folder = os.path.join(write_to_file, fold, subject)
        os.makedirs(subject_folder, exist_ok=True)
        torch.save(graph, os.path.join(subject_folder, f"{subject}.graph"))
        torch.save(voxels, os.path.join(subject_folder, f"{subject}.map"))

    return graph, centroids


def extract_patches(
        multi_channel_volume,
        patch_size=(8, 8, 8),
        stride=(8, 8, 8),
        min_nonzero_ratio=0.1
    ):
    """
    Extracts patches from a multichannel 4D MRI image.
    Args:
        multi_channel_volume (torch.Tensor or numpy.ndarray): The 4D input image (C, D, H, W).
        patch_size (tuple): Size of each patch (depth, height, width).
        stride (tuple): Stride for patch extraction.
        min_nonzero_ratio (float): Minimum ratio of non-zero voxels to include a patch.

    Returns:
        patches (list): List of patches, each a numpy array of shape (C, patch_d, patch_h, patch_w).
        patch_coordinates (list): List of patch coordinates (start_d, start_h, start_w).
    """
    if isinstance(multi_channel_volume, torch.Tensor):
        volume = multi_channel_volume.cpu().numpy()
    else:
        volume = np.array(multi_channel_volume)

    C, D, H, W = volume.shape
    patch_d, patch_h, patch_w = patch_size
    stride_d, stride_h, stride_w = stride

    patches = []
    patch_coordinates = []

    for d in range(0, D - patch_d + 1, stride_d):
        for h in range(0, H - patch_h + 1, stride_h):
            for w in range(0, W - patch_w + 1, stride_w):
                patch = volume[:, d:d+patch_d, h:h+patch_h, w:w+patch_w]

                # Check if patch has sufficient non-zero content
                if np.count_nonzero(patch) / patch.size >= min_nonzero_ratio:
                    patches.append(patch)
                    patch_coordinates.append((d, h, w))

    return patches, patch_coordinates


def get_patch_voxel_indices(coords, patch_size):
    """
    Get voxel indices for a patch.
    Args:
        coords (tuple): Patch coordinates (start_d, start_h, start_w).
        patch_size (tuple): Patch size (patch_d, patch_h, patch_w).
    Returns:
        voxel_indices (numpy.ndarray): Array of voxel indices in the patch.
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    d_coords, h_coords, w_coords = np.meshgrid(
        np.arange(start_d, start_d + patch_d),
        np.arange(start_h, start_h + patch_h),
        np.arange(start_w, start_w + patch_w),
        indexing='ij'
    )
    return np.column_stack([d_coords.flatten(), h_coords.flatten(), w_coords.flatten()])


def get_patch_centroid(coords, patch_size):
    """
    Calculate the centroid of a patch.
    Args:
        coords (tuple): Patch coordinates (start_d, start_h, start_w).
        patch_size (tuple): Patch size (patch_d, patch_h, patch_w).
    Returns:
        centroid (numpy.ndarray): Centroid coordinates.
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    return np.array([
        start_d + (patch_d - 1) / 2.0,
        start_h + (patch_h - 1) / 2.0,
        start_w + (patch_w - 1) / 2.0
    ])


def get_patch_labels(coords, patch_size, truth):
    """
    Computes the label for a patch by majority voting using np.bincount.
    This method mirrors the logic of finding a label for a supervoxel.

    Args:
        coords (tuple): Patch coordinates (start_d, start_h, start_w).
        patch_size (tuple): Patch size (patch_d, patch_h, patch_w).
        truth (torch.Tensor or numpy.ndarray): The ground truth label volume,
            typically 4D (C, D, H, W). The labels are expected to be integers
            corresponding to BraTS classes.

    Returns:
        patch_label (int): The majority label in the patch, corresponding to one of the
            BraTS-2023 classes:
            - 0: Non-tumorous area (NT)
            - 1: Necrotic tumor core (NCR)
            - 2: Peritumoral edematous/invaded tissue (ED)
            - 3: GD-enhancing tumor (ET)
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    
    if isinstance(truth, torch.Tensor):
        truth_np = truth.cpu().numpy()
    else:
        truth_np = np.asanyarray(truth) # Use asanyarray to avoid copying if already numpy

    # 1. Extract the 3D label data for the patch from the first channel
    patch_volume_labels = truth_np[0,
        start_d : start_d + patch_d,
        start_h : start_h + patch_h,
        start_w : start_w + patch_w
    ]

    # 2. Flatten the 3D patch into a 1D array of voxel labels
    voxel_labels = patch_volume_labels.flatten()

    # Edge case: if the patch is empty, return background label
    if voxel_labels.size == 0:
        return 0

    # 3. Use np.bincount to count occurrences of each label and find the most frequent one (the majority)
    # This is the exact same logic as in your get_node_labels function.
    # We ensure the type is integer, as required by np.bincount.
    majority_label = np.bincount(voxel_labels.astype(np.int64)).argmax()
    
    return majority_label


def get_node_edges(centroids, k=10):
    """
    Computes graph edges using k-nearest neighbors on node centroids.
    Args:
        centroids (numpy.ndarray): List of node centroids.
        k (int): Number of neighbor edges to keep for each node.
    Returns:
        edges (numpy.ndarray): The graph edges in shape (num_edges, 2).
    """
    num_nodes = len(centroids)
    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)
    
    # Ensure k is not greater than the number of possible neighbors
    k = min(k, num_nodes - 1)
    
    if k <= 0:
        return np.array([]).reshape(0, 2)

    # Find the k+1 nearest neighbors (the first one is the node itself)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(centroids)
    _, indices = nn.kneighbors(centroids)

    # Create edge list
    rows = np.repeat(np.arange(num_nodes), k)
    cols = indices[:, 1:].flatten()
    edges = np.stack([rows, cols], axis=1)

    return edges