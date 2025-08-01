import torch
import torch.nn as nn
import types
from sklearn.decomposition import IncrementalPCA
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import gc
import pickle
import os
from copy import deepcopy
from CLAP import get_audio_features
import torch.nn.functional as F

class ResiDual(nn.Module):
    """
    Apply an anisotropic scaling of a transformer residual stream unit representations (attention units)
    regulated by a learnable parameter (as in the paper https://arxiv.org/pdf/2411.00246)
    """

    def __init__(self, pca_basis, pca_mean, n_components=None):
        super().__init__()
        D = pca_basis.shape[0]
        self.n_components = n_components or D

        self.register_buffer("mean", pca_mean) # [D]
        self.register_buffer("basis", pca_basis[:self.n_components])  # [D, D]
        self.learnable = nn.Parameter(torch.ones(self.n_components))

    def forward(self, x):
        """
        x: input residual unit vector (shape [B, N, D])
        """
        x_centered = x - self.mean
        x_proj = torch.matmul(x_centered, self.basis.T)
        x_scaled = x_proj * self.learnable
        x_out = torch.matmul(x_scaled, self.basis)

        return x_out
    

def patch_block_with_residual(block, residual):
    """Inject ResiDual inside a SwinTransformerBlock (https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/htsat.py)"""

    orig_attn = block.attn
    orig_mlp = block.mlp
    norm1 = block.norm1
    norm2 = block.norm2
    drop_path = block.drop_path
    attn_mask = block.attn_mask
    shift_size = block.shift_size
    window_size = block.window_size
    input_resolution = block.input_resolution

    def patched_forward(self, x):
        H, W = input_resolution
        B, L, C = x.shape
        shortcut = x
        x = norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)

        # attention
        attn_windows, attn = orig_attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, H, W)

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        
        x = residual(x) # apply ResiDual after attention and before MLP

        x = shortcut + drop_path(x)
        x = x + drop_path(orig_mlp(norm2(x)))

        return x, attn

    block.forward = types.MethodType(patched_forward, block)


def compute_pca_components(model, dataloader, target_layer, n_components=None, max_batches=None, 
                           save_path=None, max_len=480000, data_filling='repeatpad', pad_or_truncate=False):
    """
     Compute PCA basis and mean for Residual in a specific Swin Transformer layer
     (apply on the attention representations collected from each block in the later)
    """    
    model.eval()
    pca_model = IncrementalPCA(n_components=n_components)

    for i,batch in enumerate(tqdm(dataloader, desc=f"Applying PCA on layer {target_layer}")):

        if max_batches and i >= max_batches:
            break

        x = batch[0] # batch_size x channels x time samples
        audio_data = quantize_tensor(x.squeeze(1)).cpu()

        func = lambda y: y
        if pad_or_truncate: func = lambda y: pad_or_truncate(y, target_len=max_len)
        audio_input = [
                get_audio_features({}, func(waveform.cpu()), max_len,
                    data_truncating='fusion' if model.enable_fusion else 'rand_trunc',
                    data_filling=data_filling,
                    audio_cfg=model.model_cfg['audio_cfg'],
                    require_grad=waveform.requires_grad
                )
                for waveform in audio_data
            ]

        # Extract attention residuals
        with torch.no_grad():
            out_dict = model.model.get_audio_output_dict(audio_input)
            res = out_dict["layers_residuals"][target_layer]  # list[Tensor of (batch_size, seq_len, dim)]

        X = res.cpu().numpy().reshape(-1, res.shape[-1])
        pca_model.partial_fit(X)

        torch.cuda.empty_cache()
        gc.collect()

    pca_results = {}
    pca_results = {
            "components": pca_model.components_,
            "mean": pca_model.mean_,
            "explained_variance": pca_model.explained_variance_,
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "n_components": pca_model.n_components_,
            "input_dim": X.shape[-1],
            "num_samples": pca_model.n_samples_seen_}

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(pca_results, f)
        print(f"PCA results saved to {save_path}")

    return pca_results
    
def load_residual(pca_path):

    with open(pca_path, "rb") as f:
        pca_results = pickle.load(f)

    components = pca_results["components"]
    mean = pca_results["mean"]

    basis = torch.tensor(components, dtype=torch.float32)
    mean = torch.tensor(mean, dtype=torch.float32)

    return ResiDual(basis, mean)

def setup_residual_htsat(model, residual, layer, blocks):
    """
    Inject ResiDual into the block locations (in a single layer) given in the input target list
    """

    # freeze everything except ResiDual scaling parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in residual.parameters():
        p.requires_grad = False

    residual.learnable.requires_grad = True

    for b in blocks:
        patch_block_with_residual(model.layers[layer].blocks[b], deepcopy(residual))
    
    return model, residual


def quantize_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    return (audio_tensor * 32767.0).to(torch.int16).to(torch.float32) / 32767.0

def pad_or_truncate(audio_tensor, target_len=480000):
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0) 
    length = audio_tensor.shape[0]
    if length > target_len:
        return audio_tensor[:target_len]
    elif length < target_len:
        return F.pad(audio_tensor, (0, target_len - length), mode='constant')
    return audio_tensor


# from https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/htsat.py

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x