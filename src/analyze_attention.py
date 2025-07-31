import torch
from CLAP import get_audio_features, int16_to_float32, float32_to_int16
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from collections import defaultdict
import numpy as np
import gc
import torch.nn.functional as F
import os
import csv


def run_PCA(clap, dataloader, num_layers, num_heads, components=None, data_filling='repeatpad', pad_or_truncate=False):
  """
  Fit an IncrementalPCA object for each layer and attention head of the given model
  on the specified data (specify the number of attention heads in each layer with a list or dict)
  """

  # initialize PCA objects
  pca_models = defaultdict(dict)
  for l in range(num_layers):
      for h in range(num_heads[l]):
          pca_models[l][h] = IncrementalPCA(n_components=components)


  for batch in tqdm(dataloader, desc="Applying PCA per layer / per attention head"):

    # apply incremental PCA on buffered data to save computation
    batch_data = defaultdict(list)
    buffers = defaultdict(list)
    BATCH_THREHSOLD = 30

    x = batch[0] # batch_size x channels x time samples

    # Extract attention
    attn = extract_attention(clap, x, data_filling=data_filling, pad_or_truncate=pad_or_truncate)
    attn_cpu = [layer.cpu() for layer in attn]

    for l, layer_attn in enumerate(attn_cpu):  # attn[layer]: list of [total_windows, heads, window_size**2, window_size**2]
          for window_attn in layer_attn:     # one window: Tensor [num_heads, window_size**2, window_size**2]
              for h in range(window_attn.shape[0]):
                  head_attn = window_attn[h]  # Tensor  [window_size**2, window_size**2]
                  flat = head_attn.flatten().numpy()  # shape: [window_size**4]
                  batch_data[(l, h)].append(flat)

    for (l, h), samples in batch_data.items():
        buffers[(l,h)].extend(samples)

        if len(buffers[(l,h)]) >= BATCH_THREHSOLD:
          X = np.stack(buffers[(l, h)])
          pca_models[l][h].partial_fit(X) # apply PCA on buffered data
          buffers[(l,h)] = []

    # Clear memory
    del out_dict, attn
    torch.cuda.empty_cache()
    gc.collect()

  return pca_models


def save_pca_results_on_file(save_dir, dataset_name, fold, pca_models):
 
  os.makedirs(save_dir, exist_ok=True)
  csv_path = os.path.join(save_dir, f"{dataset_name}-fold{fold}.csv")

  with open(csv_path, mode="w", newline="") as file:
      
      writer = csv.writer(file)
      writer.writerow([
          "layer", "head", "component_index",
          "explained_variance",
          "explained_variance_ratio",
          "participation_ratio",
          "intrinsic_dim"
      ])

      # values for each layer/head/component
      for layer_idx, layer in pca_models.items():
          for head_idx, pca in layer.items():
              if not hasattr(pca, "explained_variance_"):
                  continue  # skip if PCA not fitted

              exp_var = pca.explained_variance_
              ratios = pca.explained_variance_ratio_
              cumsum = ratios.cumsum()
              intrinsic_dim = (cumsum < 0.99).sum() + 1
              pr = (exp_var.sum() ** 2) / np.sum(exp_var ** 2)

              for i, (ev, ratio) in enumerate(zip(exp_var, ratios)):
                  writer.writerow([
                      layer_idx,
                      head_idx,
                      i,
                      ev,
                      ratio,
                      pr if i == 0 else "",  # write once per head
                      intrinsic_dim if i == 0 else ""
                  ])


def load_pca_csv_results(path):

    results = defaultdict(lambda: {
        "explained_variance": [],
        "explained_variance_ratio": [],
        "participation_ratio": None,
        "intrinsic_dim": None
    })

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])

            key = (layer, head)
            results[key]["explained_variance"].append(float(row["explained_variance"]))
            results[key]["explained_variance_ratio"].append(float(row["explained_variance_ratio"]))

            # if participation_ratio is non-empty and not set yet
            pr = row.get("participation_ratio", "")
            if pr and results[key]["participation_ratio"] is None:
                results[key]["participation_ratio"] = float(pr)

            dim = row.get("intrinsic_dim", "")
            if dim and results[key]["intrinsic_dim"] is None:
                results[key]["intrinsic_dim"] = float(dim)

    return results


def extract_attention(clap, X, max_len=480000, data_filling='repeatpad', pad_or_truncate=False):
    """
    Return attention weights for each layer of the given model 
    for the specified data X of shape [batch_size, 1, time_samples]
    """
    audio_data = quantize_tensor(X.squeeze(1)).cpu()
    func = lambda y: y
    if pad_or_truncate: func = lambda y: pad_or_truncate(y, target_len=max_len)

    audio_input = [
          get_audio_features({}, func(waveform.cpu()), max_len,
              data_truncating='fusion' if clap.enable_fusion else 'rand_trunc',
              data_filling=data_filling,
              audio_cfg=clap.model_cfg['audio_cfg'],
              require_grad=waveform.requires_grad
          )
          for waveform in audio_data
      ]

    # Extract attention
    with torch.no_grad():
        out_dict = clap.model.get_audio_output_dict(audio_input)
        attn = out_dict["layers_attention"]  # list of Tensors of shape [batch_size * num_windows, num_heads, window_size**2, window_size**2)]

    return attn


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