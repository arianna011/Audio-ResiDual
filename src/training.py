import torch
import torch.nn as nn
from tqdm import tqdm
from CLAP import get_audio_features
from src import quantize_tensor, pad_or_truncate, load_residual, setup_residual_htsat
import numpy as np
import os
from pathlib import Path
import wandb


def train_one_epoch_zero_shot(model, dataloader, text_embeddings, optimizer, criterion, device):
    """
    Train ResiDual via zero-shot supervision using fixed label text embeddings
    """

    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, true_labels in tqdm(dataloader, desc="Training (zero-shot)"):
        optimizer.zero_grad()

        audio_data = x.squeeze(1).to(device)
        audio_embeds = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True) # batch_size x D
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


def evaluate(model, dataloader, text_embeddings, criterion, device):
    """
    Evaluate ResiDual on zero-shot classification with fixed label text embeddings
    """

    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, true_labels in tqdm(dataloader, desc="Evaluating (zero-shot)"):

            audio_data = quantize_tensor(x.squeeze(1)).cpu().numpy()
            audio_embeds = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False) # batch_size x D
            audio_embeds = torch.tensor(audio_embeds).to(device).float()

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


def train_with_config(config, clap, dataset_name, folds, text_embeds, pca_path, project_name="residual-clap"):
    """
    Train ResiDual with a Weights&Bias sweep with K-fold cross validation

    Params:
        config: W&B sweep configuration
        clap: entire CLAP Module
        dataset_name: id of the considered dataset
        folds: ordered list of (train_loader, val_loader) dataloaders for the given dataset
        text_embeds: embeddings of the class labels
        pca_path: path to the directory containing files storing PCA basis and mean
    """

    lr = config.learning_rate
    epochs = config.epochs
    layers = config.inject_layers
    eval_fold = config.eval_fold

    layers_str = '_'.join(map(str, layers))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"lr={lr}_ep={epochs}_L={layers_str}_evalfold={eval_fold}"
    
    wandb.init(project=project_name, name=run_name, config={
        "dataset": dataset_name,
        "fold": eval_fold,
        "learning_rate": lr,
        "epochs": epochs,
        "inject_layers": layers,
        "inject_layers_str": layers_str
    })

    train_loader, val_loader = folds[eval_fold]
    pca_files = {l: os.path.join(pca_path, dataset_name, f"layer_{l}_evalfold_{eval_fold}") for l in layers}
   
    # load frozen CLAP and inject new ResiDual unit
    audio_encoder = clap.model.audio_branch
    new_htsat, residuals = setup_residual_htsat(audio_encoder, pca_files, layers)
    clap.model.audio_branch = new_htsat

    # setup training
    optimizer = torch.optim.Adam([res.learnable for res in residuals.values()], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0

    for epoch in range(epochs):
            
        train_loss, train_acc = train_one_epoch_zero_shot(clap, train_loader, text_embeds, optimizer, criterion, device)
        val_loss, val_acc = evaluate(clap, val_loader, text_embeds, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc

        wandb.log({
                "fold": eval_fold,
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc
            }, step=epoch+1)

        # log learnable parameter values
        for layer_id, residual in residuals.items():
                wandb.log({f'learnable/layer_{layer_id}/fold_{eval_fold}':  wandb.Histogram(residual.learnable.detach().cpu().numpy())}, step=epoch+1)

    
    wandb.run.summary[f"fold_{eval_fold}_best_val_accuracy"] = best_acc
    for layer_id, residual in residuals.items():
        wandb.run.summary[f"final_learnable/layer_{layer_id}/fold_{eval_fold}"] = wandb.Histogram(residual.learnable.detach().cpu().numpy())  

    print(f"Fold {eval_fold} - Best Val Acc: {best_acc:.4f}")
    wandb.finish()


