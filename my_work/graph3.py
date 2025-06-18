# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from monai.networks.blocks import Convolution
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers.factories import Act, Norm
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
)
from monai.utils import deprecated_arg


__all__ = [
    "ViT",
    "image_to_graph_vit",
    "extract_patches",
    "get_node_edges",
    "get_patch_voxel_indices",
    "get_patch_centroid",
    "get_patch_labels",
]


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        pos_embed_type: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        # my add
        name="ViT",
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            proj_type (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed_type (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), proj_type='conv', pos_embed_type='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), proj_type='conv', pos_embed_type='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), proj_type='conv', pos_embed_type='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

        self.name = name

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out

def image_to_graph_vit(
    data: dict,
    vit_model: ViT,  
    patch_size=(8, 8, 8),
    k=10,
    num_important_features=20,
    write_to_file: str | None = None,
    saved_path: str | None = None,
    saved_model: str | None = None,
):
    """
    Constructs a graph from an image using a plain ViT model for feature extraction.

    This function feeds an entire image to a ViT, uses its output token sequence as node
    features, reduces their dimensionality with PCA, and then builds the graph.
    It does NOT require the ViTAutoEncoder3D or a decoder.

    Args:
        data (dict): Dictionary containing 'image' and 'label' paths.
        vit_model (ViT): The pre-trained plain ViT model (must be initialized with classification=False).
        patch_size (tuple): The size of patches corresponding to each node in the ViT grid.
        k (int): Number of nearest neighbors for graph edge construction.
        num_important_features (int): Target number of features after PCA reduction.
        write_to_file (str, optional): Path to save the graph and map files.
        saved_path (str, optional): Path to the directory with saved model weights.
        saved_model (str, optional): Specific model file name to load.

    Returns:
        tuple: A tuple containing:
            - graph (torch_geometric.data.Data or None): The constructed graph object.
            - centroids (numpy.ndarray or None): The 3D coordinates of the graph nodes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_pipeline = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image"], spatial_size=(112, 112, 72), mode="trilinear", align_corners=False),
            Resized(keys=["label"], spatial_size=(112, 112, 72), mode="nearest"),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    try:
        processed_data = transform_pipeline(data)
    except Exception as e:
        print(f"Error processing data for subject {data.get('subject', 'N/A')}: {e}")
        return None, None

    image_volume = processed_data["image"]
    label_volume = processed_data["label"]

    image_tensor = torch.tensor(image_volume, dtype=torch.float).unsqueeze(0).to(device)

    # Note: Model loading logic might need adjustment if model names don't match
    if saved_path:
        model_name_key = "ViT_4CLASS"
        model_files = [n for n in os.listdir(saved_path) if model_name_key.upper() in n and n.endswith(".pth")]
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint found for {model_name_key} in {saved_path}")
        last_model = saved_model if saved_model else sorted(model_files)[-1]
        vit_model.load_state_dict(torch.load(os.path.join(saved_path, last_model), map_location=device))

    vit_model.to(device)
    vit_model.eval()

    with torch.no_grad():
        node_features_raw, _ = vit_model(image_tensor)

    # The raw output from ViT (with classification=False) is a 3D tensor:
    # (batch_size, num_patches, hidden_size)
    if node_features_raw.dim() == 3:
        # Remove the batch dimension to get (num_patches, hidden_size)
        all_node_features = node_features_raw.squeeze(0)
    else:
        raise ValueError(f"Unsupported shape for node features from ViT: {node_features_raw.shape}")

    # --- Reduce feature dimension using PCA ---
    original_feature_dim = all_node_features.shape[1]
    if num_important_features > original_feature_dim:
        print(
            f"Warning: Requested {num_important_features} features, but only {original_feature_dim} are available. "
            f"Using {original_feature_dim}."
        )
        num_important_features = original_feature_dim

    features_np = all_node_features.cpu().numpy()
    pca = PCA(n_components=num_important_features)
    reduced_features_np = pca.fit_transform(features_np)
    node_features = torch.tensor(reduced_features_np, dtype=torch.float, device=device)

    # --- Generate spatial and label info for all nodes (this logic is unchanged) ---
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

    # --- Build the final graph using all nodes but with reduced features ---
    graph = Data(
        x=node_features,
        edge_index=torch.tensor(get_node_edges(centroids=centroids, k=k).T, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
    )

    if write_to_file and "subject" in data:
        subject = data["subject"]
        subject_folder = os.path.join(write_to_file, "graphs", subject)
        os.makedirs(subject_folder, exist_ok=True)
        torch.save(graph, os.path.join(subject_folder, f"{subject}.graph"))
        torch.save(voxels, os.path.join(subject_folder, f"{subject}.map"))

    return graph, centroids

def extract_patches(multi_channel_volume, patch_size=(8, 8, 8), stride=(8, 8, 8), min_nonzero_ratio=0.1):
    if isinstance(multi_channel_volume, torch.Tensor):
        volume = multi_channel_volume.cpu().numpy()
    else:
        volume = np.array(multi_channel_volume)

    C, D, H, W = volume.shape
    patch_d, patch_h, patch_w = patch_size
    stride_d, stride_h, stride_w = stride
    patches, patch_coordinates = [], []

    for d in range(0, D - patch_d + 1, stride_d):
        for h in range(0, H - patch_h + 1, stride_h):
            for w in range(0, W - patch_w + 1, stride_w):
                patch = volume[:, d : d + patch_d, h : h + patch_h, w : w + patch_w]
                if np.count_nonzero(patch) / patch.size >= min_nonzero_ratio:
                    patches.append(patch)
                    patch_coordinates.append((d, h, w))
    return patches, patch_coordinates


def get_patch_voxel_indices(coords, patch_size):
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    d_coords, h_coords, w_coords = np.meshgrid(
        np.arange(start_d, start_d + patch_d),
        np.arange(start_h, start_h + patch_h),
        np.arange(start_w, start_w + patch_w),
        indexing="ij",
    )
    return np.column_stack([d_coords.flatten(), h_coords.flatten(), w_coords.flatten()])


def get_patch_centroid(coords, patch_size):
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    return np.array(
        [start_d + (patch_d - 1) / 2.0, start_h + (patch_h - 1) / 2.0, start_w + (patch_w - 1) / 2.0]
    )


def get_patch_labels(coords, patch_size, truth):
    start_d, start_h, start_w = coords
    patch_d, patch_h, patch_w = patch_size
    truth_np = truth.cpu().numpy() if isinstance(truth, torch.Tensor) else np.asanyarray(truth)
    patch_volume_labels = truth_np[0, start_d : start_d + patch_d, start_h : start_h + patch_h, start_w : start_w + patch_w]
    voxel_labels = patch_volume_labels.flatten()
    if voxel_labels.size == 0:
        return 0
    return np.bincount(voxel_labels.astype(np.int64)).argmax()


def get_node_edges(centroids, k=10):
    num_nodes = len(centroids)
    if num_nodes <= 1:
        return np.array([]).reshape(0, 2)
    k = min(k, num_nodes - 1)
    if k <= 0:
        return np.array([]).reshape(0, 2)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(centroids)
    _, indices = nn.kneighbors(centroids)
    rows = np.repeat(np.arange(num_nodes), k)
    cols = indices[:, 1:].flatten()
    return np.stack([rows, cols], axis=1)