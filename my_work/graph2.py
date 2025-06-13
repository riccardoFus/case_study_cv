"""
A set of functions for graph construction and manipulation using patch embedding
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.helpers.utils import get_device, get_config
from src.modules.training import predict_ae
from src.modules.preprocessing import get_transformations


__all__ = ['image_to_graph_2']


def image_to_graph_2(
        data,
        patch_size = (8, 8, 8),
        stride = None,
        model = None,
        saved_path = None,
        percentiles = [10, 25, 50, 75, 90],
        k = 10,
        write_to_file = None,
        min_nonzero_ratio = 0.1
    ):
    """
    Converts a Dataloader entry to graph structure using patch embedding.
    Args:
        data (dict): an example returned from Dataloader class.

        patch_size (tuple): size of each patch (depth, height, width).
        stride (tuple | None): stride for patch extraction. If None, uses patch_size (non-overlapping).

        model (src.models.autoencoder3d.AutoEncoder3D | None): the AutoEncoder3D model.
            If None the resulting graph will contain only the percentile intensity features.
        saved_path (str | None): folder path where the pretrained model is saved.

        percentiles (list): list of intensities percentiles to compute.
        k (int): number of edges neighbors to keep.
        write_to_file (str | None): Whether save the resulting graph to file. If not None
            a valid folder path where to save files must be specified.
        min_nonzero_ratio (float): minimum ratio of non-zero voxels in patch to include it.

    Returns:
        graph (torch_geometric.data.data.Data): the graph representing a 4D multichannel image.
        centroids (numpy.ndarray): the graph nodes centroids.
    """
    base_image_transform, _, autoencoder_eval_transform, _, _ = get_transformations()
    image_trans = base_image_transform(data)
    
    if stride is None:
        stride = patch_size
    
    patches, patch_coordinates = extract_patches(
        multi_channel_volume = image_trans['image'],
        patch_size = patch_size,
        stride = stride,
        min_nonzero_ratio = min_nonzero_ratio
    )
    
    if not model is None:
        _, latent_maps, _ = predict_ae(
            model = model,
            data = data,
            transforms = autoencoder_eval_transform,
            device = 'cpu', # get_device(), # 'mps' not supported
            saved_path = saved_path
        )
    
    node_features, centroids, labels, voxels = [], [], [], []
    
    for i, (patch, coords) in enumerate(zip(patches, patch_coordinates)):
        # Extract percentile features from patch
        patch_features = get_patch_percentiles_features(
            patch = patch,
            percentiles = percentiles
        )
        
        # Get patch mask for label extraction
        patch_mask = get_patch_mask(coords, image_trans['image'].shape[1:])
        
        if not model is None:
            # Extract latent features for this patch
            latent_features = get_patch_latent_features(
                coords = coords,
                latent_maps = latent_maps,
                patch_size = patch_size
            )
            node_features.append(np.concatenate([patch_features, latent_features]))
        else:
            node_features.append(patch_features)
        
        # Store patch information
        voxels.append(get_patch_voxel_indices(coords, patch_size))
        centroids.append(get_patch_centroid(coords, patch_size))
        labels.append(get_patch_labels(coords, patch_size, image_trans['label']))
    
    centroids = np.array(centroids)
    graph = Data(
        x = torch.tensor(node_features, dtype=torch.float),
        edge_index = torch.tensor(get_node_edges(centroids = centroids, k = k).T, dtype=torch.long),
        y = torch.tensor(labels, dtype=torch.float)
    )

    # Save data to file
    if write_to_file:
        fold = get_config().get('GRAPH_FOLDER')
        if not os.path.isdir(os.path.join(write_to_file, fold)):
            os.makedirs(os.path.join(write_to_file, fold))
        if not os.path.isdir(os.path.join(write_to_file, fold, data['subject'])):
            os.makedirs(os.path.join(write_to_file, fold, data['subject']))
        torch.save(graph, os.path.join(write_to_file, fold, data['subject'], (data['subject'] + '.graph')))
        torch.save(voxels, os.path.join(write_to_file, fold, data['subject'], (data['subject'] + '.map')))
    
    return graph, centroids


def extract_patches(
        multi_channel_volume,
        patch_size = (8, 8, 8),
        stride = (8, 8, 8),
        min_nonzero_ratio = 0.1
    ):
    """
    Extract patches from a multichannel 4D MRI image.
    Args:
        multi_channel_volume (numpy.ndarray): the 4D multichannel input image (C, D, H, W).
        patch_size (tuple): size of each patch (depth, height, width).
        stride (tuple): stride for patch extraction.
        min_nonzero_ratio (float): minimum ratio of non-zero voxels in patch to include it.
    
    Returns:
        patches (list): list of patches, each with shape (C, patch_d, patch_h, patch_w).
        patch_coordinates (list): list of patch coordinates (start_d, start_h, start_w).
    """
    if isinstance(multi_channel_volume, torch.Tensor):
        volume = multi_channel_volume.numpy()
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
                nonzero_ratio = np.count_nonzero(patch) / patch.size
                if nonzero_ratio >= min_nonzero_ratio:
                    patches.append(patch)
                    patch_coordinates.append((d, h, w))
    
    return patches, patch_coordinates


def get_patch_percentiles_features(
        patch,
        percentiles = [10, 25, 50, 75, 90]
    ):
    """
    Computes the percentiles features for a patch.
    Args:
        patch (numpy.ndarray): patch with shape (C, D, H, W).
        percentiles (list): list of intensities percentiles to compute.
    Returns:
        patch_features (numpy.ndarray): the percentiles features extracted from patch.
    """
    patch_features = []
    C = patch.shape[0]
    
    for c in range(C):
        channel_patch = patch[c]
        # Only consider non-zero voxels for percentile calculation
        nonzero_voxels = channel_patch[channel_patch > 0]
        if len(nonzero_voxels) > 0:
            channel_percentiles = np.percentile(nonzero_voxels, percentiles)
        else:
            channel_percentiles = np.zeros(len(percentiles))
        patch_features.extend(channel_percentiles)
    
    return np.array(patch_features)


def get_patch_latent_features(
        coords,
        latent_maps,
        patch_size,
        latent_stride = (2, 2, 2)
    ):
    """
    Extract latent features for a patch from the autoencoder latent maps.
    Args:
        coords (tuple): patch coordinates (start_d, start_h, start_w).
        latent_maps (torch.Tensor): latent feature maps from autoencoder.
        patch_size (tuple): original patch size.
        latent_stride (tuple): stride used in latent space.
    Returns:
        latent_features (numpy.ndarray): extracted latent features.
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    
    # Convert to latent space coordinates
    latent_start_d = start_d // latent_stride[0]
    latent_start_h = start_h // latent_stride[1] 
    latent_start_w = start_w // latent_stride[2]
    
    latent_end_d = min(latent_maps.shape[1], (start_d + patch_d) // latent_stride[0])
    latent_end_h = min(latent_maps.shape[2], (start_h + patch_h) // latent_stride[1])
    latent_end_w = min(latent_maps.shape[3], (start_w + patch_w) // latent_stride[2])
    
    # Extract latent patch
    latent_patch = latent_maps[:,
        latent_start_d:latent_end_d,
        latent_start_h:latent_end_h,
        latent_start_w:latent_end_w
    ]
    
    # Resize to fixed size and extract features
    fixed_size = (3, 3, 3)
    if latent_patch.numel() > 0:
        latent_patch_resized = F.adaptive_avg_pool3d(latent_patch.unsqueeze(0), fixed_size).squeeze(0)
        features = np.array(latent_patch_resized).mean(axis=0).flatten()
        
        # Apply SVD for dimensionality reduction
        if features.size >= 9:
            _, S, _ = np.linalg.svd(features.reshape(-1, 3) if features.size % 3 == 0 else features.reshape(9, -1))
            latent_features = np.concatenate([S, features])
        else:
            latent_features = features
    else:
        latent_features = np.zeros(30)  # Default size
    
    return latent_features


def get_patch_mask(coords, volume_shape):
    """
    Create a binary mask for the patch location.
    Args:
        coords (tuple): patch coordinates (start_d, start_h, start_w).
        volume_shape (tuple): shape of the original volume (D, H, W).
    Returns:
        mask (numpy.ndarray): binary mask indicating patch location.
    """
    D, H, W = volume_shape
    mask = np.zeros((D, H, W), dtype=bool)
    start_d, start_h, start_w = coords
    
    # This would need patch_size to be properly implemented
    # For now, return a simple point mask
    if start_d < D and start_h < H and start_w < W:
        mask[start_d, start_h, start_w] = True
    
    return mask


def get_patch_voxel_indices(coords, patch_size):
    """
    Get voxel indices for a patch.
    Args:
        coords (tuple): patch coordinates (start_d, start_h, start_w).
        patch_size (tuple): patch size (patch_d, patch_h, patch_w).
    Returns:
        voxel_indices (numpy.ndarray): array of voxel indices in the patch.
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    
    # Generate all voxel coordinates in the patch
    d_coords, h_coords, w_coords = np.meshgrid(
        np.arange(start_d, start_d + patch_d),
        np.arange(start_h, start_h + patch_h),
        np.arange(start_w, start_w + patch_w),
        indexing='ij'
    )
    
    voxel_indices = np.column_stack([
        d_coords.flatten(),
        h_coords.flatten(),
        w_coords.flatten()
    ])
    
    return voxel_indices


def get_patch_centroid(coords, patch_size):
    """
    Calculate the centroid of a patch.
    Args:
        coords (tuple): patch coordinates (start_d, start_h, start_w).
        patch_size (tuple): patch size (patch_d, patch_h, patch_w).
    Returns:
        centroid (numpy.ndarray): centroid coordinates.
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    
    centroid = np.array([
        start_d + patch_d / 2.0,
        start_h + patch_h / 2.0,
        start_w + patch_w / 2.0
    ])
    
    return centroid


def get_patch_labels(coords, patch_size, truth):
    """
    Computes the labels for a patch by majority voting.
    Args:
        coords (tuple): patch coordinates (start_d, start_h, start_w).
        patch_size (tuple): patch size (patch_d, patch_h, patch_w).
        truth (numpy.ndarray): the 4D multichannel label image.
    Returns:
        patch_label (int): the majority label in the patch.
    """
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    
    # Extract patch from ground truth
    patch_labels = truth[0, 
        start_d:start_d + patch_d,
        start_h:start_h + patch_h,
        start_w:start_w + patch_w
    ]
    
    # Get majority label
    unique_labels, counts = np.unique(patch_labels.flatten(), return_counts=True)
    majority_label = unique_labels[np.argmax(counts)]
    
    return int(majority_label)


def get_node_edges(centroids, k=10):
    """
    Computes the graph edges using k-nearest neighbors.
    Args:
        centroids (numpy.ndarray): the graph centroid list.
        k (int): number of neighbor edges to keep.
    Returns:
        edges (numpy.ndarray): the graph edges.
    """
    if len(centroids) <= k:
        k = len(centroids) - 1
    
    if k <= 0:
        return np.array([]).reshape(0, 2)
    
    _, indices = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids).kneighbors(centroids)
    edges = np.array([[i, j] for i in range(indices.shape[0]) for j in indices[i, 1:]])
    
    return edges