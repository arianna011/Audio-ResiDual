import torch
from torch import nn
from tqdm import tqdm
import os
import gc
import torch.nn.functional as F
import numpy as np

class HTSATLinearClassifier(nn.Module):
    """
    Frozen HTSAT audio encoder with a trainable linear head for classification
    """

    def __init__(self, clap, n_classes, feat_dim=512):
        super().__init__()
        self.clap = clap
        self.feat_dim = feat_dim
        self.n_classes = n_classes

        for p in self.clap.parameters():
            p.requires_grad = False
        
        self.classifier = nn.Linear(self.feat_dim, self.n_classes)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, device):
        audio_data = x.squeeze(1).to(device)
        audio_embeds = self.clap.get_audio_embedding_from_data(x = audio_data, use_tensor=True) # batch_size x D
        audio_embeds = audio_embeds.to(device).float()
        logits = self.classifier(audio_embeds)
        return logits
    

def train_linear_head_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, true_labels in tqdm(dataloader, desc="Training (supervised)"):
        
        optimizer.zero_grad()
        logits = model(x, device) 
        loss = criterion(logits, true_labels.to(device))
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1).cpu()
        correct += (preds == true_labels).sum().item()
        total += x.size(0)
        total_loss += loss.item() * x.size(0)

    acc = correct / total
    return total_loss / total, acc


def train_and_eval_linear_head(clap, dataset_name, folds, n_classes, save_dir, lr=0.01, epochs=10):
    """
    Train the linear projection on HTSAT output with K-fold Cross-Validation
    and save classification predictions on file
    """

    save_dir = os.path.join(save_dir, dataset_name, "Linear")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (train_load, val_load) in enumerate(folds):

        print(f'===== Eval fold {i} =====')

        save_file = os.path.join(save_dir, f'evalfold_{i}.npz')
        model = HTSATLinearClassifier(
            clap=clap,
            n_classes=n_classes).to(device)
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(epochs):
            print(f"=== Epoch {ep} ===")
            loss, acc = train_linear_head_one_epoch(model, train_load, optimizer, criterion, device)
            print(f"Train loss: {loss}, Train accuracy: {acc}")

        preds, targs, similarities = eval_linear_head(model, val_load, device)
        np.savez_compressed(
                save_file,
                similarities = similarities,
                predictions = np.array(preds),
                targets = np.array(targs)
        )
        
        torch.cuda.empty_cache()
        gc.collect()

        

def eval_linear_head(model, dataloader, device):
    """
    Evaluate the input model on supervised classification

    Returns:
     - all predictions
     - all targets 
     - all computed logits as similarities
    """

    model.eval()
    all_preds = []
    all_targets = []
    all_similarities = []

    with torch.no_grad():
        for x, true_labels in tqdm(dataloader, desc="Evaluating (supervised)"):

            logits = model(x, device)
            preds = logits.argmax(dim=-1).cpu()
    
            all_preds.extend(preds.tolist())
            all_targets.extend(true_labels.tolist())
            sims = F.softmax(logits, dim=-1)
            all_similarities.append(sims)

    full_similarities = torch.cat(all_similarities, dim=0).cpu().numpy()
    return all_preds, all_targets, full_similarities

