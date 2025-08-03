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


def evaluate(model, dataloader, text_embeddings, criterion, device, 
            max_len=480000, data_filling='repeatpad', pad_or_truncate=False):
    """
    Evaluate ResiDual on zero-shot classification with fixed label text embeddings
    """

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, true_labels in tqdm(dataloader, desc="Evaluating (zero-shot)"):

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
            
            out_dict = model.model.get_audio_output_dict(audio_input)
            audio_embeds = out_dict["embedding"] # batch_size x D
            audio_embeds = audio_embeds.to(device).float()

            # compute similarities between audio and text embeddings
            similarities = torch.matmul(audio_embeds, text_embeddings.T.to(device))  # batch_size x num_classes
            loss = criterion(similarities, true_labels.to(device))
            preds = similarities.argmax(dim=-1).cpu()
            correct += (preds == true_labels).sum().item()
            total += x.size(0)
            total_loss += loss.item() * x.size(0)

    # return average loss and accuracy
    acc = correct / total
    return total_loss / total, acc


def train_with_config(config, clap, dataset_name, folds, text_embeds, pca_path):
    """
    Train ResiDual with a Weights&Bias sweep

    Params:
        config: W&B sweep configuration
        clap: entire CLAP Module
        dataset_name: id of the considered dataset
        folds: train and validation dataloaders for each fold of the considered dataset
        text_embdes: embeddings of the class labels
        pca_path: path to the directory containing files storing PCA basis and mean
    """

    lr = config.learning_rate
    epochs = config.epochs
    layers = config.inject_layers
    layers_str = '_'.join(map(str, layers))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"lr{lr}_ep{epochs}_L{layers_str}"
    wandb.run.name = run_name
    wandb.config.inject_layers_str = layers_str
    
    fold_accuracies = []
    for fold_idx, (train_loader, val_loader) in enumerate(folds):

        best_acc = 0.0
        pca_file = os.path.join(pca_path, f"{dataset_name}-fold{fold_idx}.csv")

        # reload frozen CLAP and inject new ResiDual unit
        residual = load_residual(pca_file).to(device)
        audio_encoder = clap.model.audio_branch
        model, residual = setup_residual_htsat(audio_encoder, residual, layers)
        clap.model.audio_branch = model

        # setup training
        optimizer = torch.optim.Adam(residual.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch_zero_shot(clap, train_loader, text_embeds, optimizer, criterion, device)
            val_loss, val_acc = evaluate(clap, val_loader, text_embeds, criterion, device)

            wandb.log({
                "fold": fold_idx + 1,
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc
            })

            if val_acc > best_acc:
                best_acc = val_acc

        wandb.run.summary[f"fold_{fold_idx+1}_best_val_accuracy"] = best_acc
        fold_accuracies.append(best_acc)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    wandb.run.summary["cv/accuracy_mean"] = mean_acc
    wandb.run.summary["cv/accuracy_std"] = std_acc

    print(f"\n{len(folds)}-fold CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    wandb.finish()


