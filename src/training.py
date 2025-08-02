import torch
import torch.nn as nn
from tqdm import tqdm
from CLAP import get_audio_features
from src import quantize_tensor, pad_or_truncate, load_residual, setup_residual_htsat
import numpy as np
import os
from pathlib import Path
import wandb


def train_one_epoch_zero_shot(model, dataloader, text_embeddings, optimizer, criterion, device, 
                              max_len=480000, data_filling='repeatpad', pad_or_truncate=False):
    """
    Train ResiDual via zero-shot supervision using fixed label text embeddings
    """

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, true_labels in tqdm(dataloader, desc="Training (zero-shot)"):
        optimizer.zero_grad()

        audio_data = quantize_tensor(x.squeeze(1)).cpu()

        # extract input features for each waveform
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
        
        with torch.no_grad():
            out_dict = model.model.get_audio_output_dict(audio_input)
            audio_embeds = out_dict["embedding"] # batch_size x D

        audio_embeds = audio_embeds.to(device).float()

        # compute similarities between audio and text embeddings
        similarities = torch.matmul(audio_embeds, text_embeddings.T.to(device))  # batch_size x num_classes

        loss = criterion(similarities, true_labels.to(device))
        loss.backward()
        optimizer.step()

        preds = similarities.argmax(dim=-1).cpu()
        correct += (preds == true_labels).sum().item()
        total += x.size(0)
        total_loss += loss.item() * x.size(0)

    # return average loss and accuracy
    acc = correct / total
    return total_loss / total, acc


def train(clap, dataloader, text_embeds, pca_path, layers, save_path, lr=0.01, epochs=20):
    """
    Train ResiDual and log with Weights&Bias

    Params:
        clap: entire CLAP Module
        dataloader: train dataset
        text_embdes: embeddings of the class labels
        pca_path: path to the file where PCA basis and mean are stored
        layers: list of layers where to inject ResiDual units
        save_path: path to the file where to store model checkpoints
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)

    model_name = "ResiDual"
    pca_name = Path(pca_path).stem
    layers_str = '_'.join(map(str, layers))
    run_id = f"{model_name}_{pca_name}_layers{layers_str}"

    wandb.init(
        project="residual-training",
        config={
            "pca_path": pca_path,
            "layers": layers,
            "learning_rate": lr,
            "epochs": epochs        
        }
    )
    wandb.run.name = run_id
    wandb.run.tags = [model_name, f"layers:{layers_str}", f"pca:{pca_name}"]

    best_acc = 0.0
    best_ckpt_path = None
    all_accuracies = []
    residual_unit = load_residual(pca_path).to(device)

    audio_encoder = clap.model.audio_branch
    # inject ResiDual
    new_model, residual_unit = setup_residual_htsat(audio_encoder, residual_unit, layers)
    clap.model.audio_branch = new_model

    # setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(residual_unit.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_zero_shot(clap, dataloader, text_embeds, optimizer, criterion, device)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc
        })
        all_accuracies.append(train_acc)

        # save best model checkpoint
        if train_acc > best_acc:
            best_acc = train_acc
            best_ckpt_path = os.path.join(save_path, run_id + "_best.pt")
            torch.save(residual_unit.state_dict(), best_ckpt_path)

            artifact = wandb.Artifact(
                f"{run_id}_best", 
                type="model",
                description=f"Best ResiDual model trained with layers {layers}, PCA: {pca_name}, accuracy: {best_acc:.4f}"
            )
            artifact.add_file(best_ckpt_path)
            wandb.log_artifact(artifact) 
            os.remove(best_ckpt_path) # remove local file


    print(f"\nFinal average accuracy after {epochs} epochs: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    wandb.finish()

