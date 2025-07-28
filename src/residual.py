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
from ..CLAP import get_audio_features

class ResiDual(nn.Module):
    """
    Apply an anisotropic scaling of a transformer residual stream unit representations (attention units)
    regulated by a learnable parameter (as in the paper https://arxiv.org/pdf/2411.00246)
    """

    def __init__(self, bases, means):
        super().__init__()
        self.num_heads = len(bases)
        self.head_dim = bases[0].shape[1]
        self.bases = nn.ParameterList([nn.Parameter(b, requires_grad=False) for b in bases]) # principal component basis for each attention head
        self.means = nn.ParameterList([nn.Parameter(m, requires_grad=False) for m in means]) # associated mean
        self.learnables = nn.ParameterList([nn.Parameter(torch.ones(b.shape[0])) for b in bases])

    def forward(self, x):
        """
        x: (B, H*W, C) with C = num_heads * head_dim
        """
        assert self.bases[0].shape[1] * len(self.bases) == x.shape[-1]

        chunks = torch.chunk(x, self.num_heads, dim=-1)
        outputs = []

        for i, x_h in enumerate(chunks):
            x_centered = x_h - self.means[i]
            x_proj = torch.matmul(x_centered, self.bases[i].T)
            x_scaled = x_proj * self.learnables[i]
            x_out = torch.matmul(x_scaled, self.bases[i])
            outputs.append(x_out)

        return torch.cat(outputs, dim=-1)
    

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
        
        x = residual(x) # apply ResiDual

        x = shortcut + drop_path(x)
        x = x + drop_path(orig_mlp(norm2(x)))

        return x, attn

    block.forward = types.MethodType(patched_forward, block)

def compute_pca_components(model, dataloader, target_layer, num_heads, n_components=None, max_batches=None, save_path=None):
    """
     Compute PCA basis and mean for Residual for each attention head in a specific Swin Transformer layer
    """    
    model.eval()

    pca_models = defaultdict(dict)
    for h in range(num_heads):
        pca_models[h] = IncrementalPCA(n_components=n_components)

    for i,batch in enumerate(tqdm(dataloader, desc=f"Applying PCA per attention head on layer {target_layer}")):

        if max_batches and i >= max_batches:
            break

        batch_data = defaultdict(list)
        buffers = defaultdict(list)
        BATCH_THREHSOLD = 30

        x = batch[0] # batch_size x channels x time samples
        audio_data = quantize_tensor(x.squeeze(1)).cpu()

        audio_input = [
                get_audio_features({}, waveform.cpu(), 480000,
                    data_truncating='fusion' if model.enable_fusion else 'rand_trunc',
                    data_filling='repeatpad',
                    audio_cfg=model.model_cfg['audio_cfg'],
                    require_grad=waveform.requires_grad
                )
                for waveform in audio_data
            ]

        # Extract attention
        with torch.no_grad():
            out_dict = model.model.get_audio_output_dict(audio_input)
            attn = out_dict["layers_attention"][target_layer]  # attn: list[batch_size * window_size**2][Tensor of (heads, window_size**2, window_size**2)]

        attn_cpu = [head.cpu() for head in attn]

        for window_attn in attn_cpu:     # one window: Tensor [heads, window_size**2, window_size**2]
            for h in range(num_heads):
                head_attn = window_attn[h]  # Tensor  [window_size**2, window_size**2]
                flat = head_attn.flatten().numpy()  # shape: [window_size**4]
                batch_data[h].append(flat)

        for h, samples in batch_data.items():
            buffers[h].extend(samples)

            if len(buffers[h]) >= BATCH_THREHSOLD:
                X = np.stack(buffers[h])
                pca_models[h].partial_fit(X) # apply PCA on buffered data
                buffers[h] = []

        # Clear memory
        del out_dict, attn
        torch.cuda.empty_cache()
        gc.collect()

    pca_results = {}
    for h in range(num_heads):
        pca_results[h] = {
            "components": pca_models[h].components_,
            "mean": pca_models[h].mean_,
            "explained_variance": pca_models[h].explained_variance_,
            "explained_variance_ratio": pca_models[h].explained_variance_ratio_,
        }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(pca_results, f)
        print(f"PCA results saved to {save_path}")

    return pca_results

    
def load_residual(pca_path):
    with open(pca_path, "rb") as f:
        pca_results = pickle.load(f)

    bases = []
    means = []

    for h in sorted(pca_results.keys()):
        components = pca_results[h]["components"]
        mean = pca_results[h]["mean"]
        bases.append(torch.tensor(components, dtype=torch.float32))
        means.append(torch.tensor(mean, dtype=torch.float32))

    return ResiDual(bases, means)

def setup_residual_htsat(model, residual, targets):
    """
    Inject ResiDual into the (layer, block) locations given in the input target list
    """

    # freeze everything except ResiDual scaling parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in residual.parameters():
        p.requires_grad = False

    for param in residual.learnables:
        param.requires_grad = True

    for (layer, block) in targets:
        patch_block_with_residual(model.layers[layer].blocks[block], residual)
    
    return model, residual


def quantize_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    return (audio_tensor * 32767.0).to(torch.int16).to(torch.float32) / 32767.0


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