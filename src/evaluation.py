import torch
from tqdm import tqdm
import os
from src import quantize_tensor, setup_residual_htsat, train_one_epoch_zero_shot
import torch.nn as nn
import gc
import numpy as np
from data_processing import DATASETS
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns



def train_and_evaluate_residual(clap, dataset_name, folds, text_embeds, pca_path, save_dir, 
                                epochs=10, lr=0.01, inject_layers=[0]):
    """
    Train and evaluate ResiDual with K-fold cross validation

    Params:
        clap: entire base CLAP Module in which to inject ResiDual
        dataset_name: id of the considered dataset
        folds: ordered list of (train_loader, val_loader) dataloaders for the given dataset
        text_embeds: embeddings of the class labels
        pca_path: path to the directory containing files storing PCA basis and mean
        save_dir: directory where to store evaluation results
        epochs: number of epochs for training
        lr: learning rate
        inject_layers: list of HTSAT encoder layers where to inject ResiDual   
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_str = '_'.join(map(str, inject_layers))

    save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for i, (train_load, val_load) in tqdm(enumerate(folds)):

        save_file = os.path.join(save_dir, f'layers_{layers_str}_evalfold_{i}.npz')
        pca_files = {l: os.path.join(pca_path, dataset_name, f"layer_{l}_evalfold_{i}") for l in inject_layers}

        audio_encoder = clap.model.audio_branch
        new_htsat, residuals = setup_residual_htsat(audio_encoder, pca_files, inject_layers)
        clap.model.audio_branch = new_htsat

        optimizer = torch.optim.Adam([res.learnable for res in residuals.values()], lr=lr)
        criterion = nn.CrossEntropyLoss()

        for e in range(epochs):
                
            print(f"=== Epoch {e} ===")
            train_loss, train_acc = train_one_epoch_zero_shot(clap, train_load, text_embeds, optimizer, criterion, device)
            print(f"Train loss: {train_loss}, Train accuracy: {train_acc}")

        preds, targs, similarities = evaluate_zero_shot(clap, val_load, text_embeds, device)
        np.savez_compressed(
            save_file,
            similarities = similarities,
            predictions = np.array(preds),
            targets = np.array(targs)
        )
        
        torch.cuda.empty_cache()
        gc.collect()


def evaluate_zero_shot(model, dataloader, text_embeddings, device):
    """
    Evaluate the input model on zero-shot classification with fixed label text embeddings

    Returns:
     - all predictions
     - all targets 
     - all computed similaritites
    """

    model.eval()
    
    all_preds = []
    all_targets = []
    all_similarities = []

    with torch.no_grad():
        for x, true_labels in tqdm(dataloader, desc="Evaluating (zero-shot)"):

            audio_data = quantize_tensor(x.squeeze(1)).cpu().numpy()
            audio_embeds = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False) # batch_size x D
            audio_embeds = torch.tensor(audio_embeds).to(device).float()

            # compute similarities between audio and text embeddings
            similarities = torch.matmul(audio_embeds, text_embeddings.T.to(device))  # batch_size x num_classes
            preds = similarities.argmax(dim=-1).cpu()
    
            all_preds.extend(preds.tolist())
            all_targets.extend(true_labels.tolist())
            all_similarities.append(similarities.cpu())

            torch.cuda.empty_cache()
            gc.collect()

    full_similarities = torch.cat(all_similarities, dim=0).numpy()
    return all_preds, all_targets, full_similarities



def visualize_eval_metrics(save_dir, dataset_name, n_folds, inject_layers, k_top=5):
    """
    Given a directory storing computed similarities, predictions and targets for audio classification for each dataset fold,
    visualize aggregated evaluation metrics like accuracy, precision, recall and confusion matrix
    """

    layers_str = ""
    if inject_layers != []:
        layers_str = '_'.join(map(str, inject_layers))
    class_names = DATASETS[dataset_name]["class_labels"]
    n_classes = len(class_names)

    per_fold = {
        "acc": [], "topk": [], "prec": [], "rec": [], "f1": []
    }
    agg_cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for i in range(n_folds):
        if layers_str: # consider ResiDual metrics
            save_file = os.path.join(save_dir, f'layers_{layers_str}_evalfold_{i}.npz')
        else: # consider linear projection metrics
            save_file = os.path.join(save_dir, f'linear_evalfold_{i}.npz')
        data = np.load(save_file)

        similarities = data["similarities"]
        predictions = data["predictions"]
        targets = data["targets"]

        y_true = np.array(targets)
        y_pred = np.array(predictions)

        acc = accuracy_score(y_true, y_pred)
        k_eff = min(k_top, similarities.shape[1])
        topk = top_k_accuracy_score(y_true, similarities, k=k_eff)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

        per_fold["acc"].append(acc)
        per_fold["topk"].append(topk)
        per_fold["prec"].append(prec)
        per_fold["rec"].append(rec)
        per_fold["f1"].append(f1)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        agg_cm += cm
    

    for k in per_fold:
        per_fold[k] = np.asarray(per_fold[k], dtype=float)

    print("== Cross-Fold Evaluation Metrics ==")
    print(f"Top-1 Accuracy:   {per_fold['acc'].mean():.4f} ± {per_fold['acc'].std(ddof=1):.4f}")
    print(f"Top-{k_top} Accuracy:  {per_fold['topk'].mean():.4f} ± {per_fold['topk'].std(ddof=1):.4f}")
    print(f"Precision: {per_fold['prec'].mean():.4f} ± {per_fold['prec'].std(ddof=1):.4f}")
    print(f"Recall:    {per_fold['rec'].mean():.4f} ± {per_fold['rec'].std(ddof=1):.4f}")
    print(f"F1:        {per_fold['f1'].mean():.4f} ± {per_fold['f1'].std(ddof=1):.4f}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(agg_cm, xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Aggregated Confusion Matrix (sum over folds)")
    plt.tight_layout()
    plt.show()