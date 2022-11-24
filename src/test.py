from pathlib import Path
from typing import List, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import wandb
from torch.nn import Dropout
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, Linear

from utils import load_config

wandb.init(project="test")

global_config = load_config("global_config.json")
data_config = load_config("data_config.json")
model_config = load_config("model_config.json")
gcn_config = model_config["gcn"]
training_config = load_config("training_config.json")

SEED = global_config["seed"]
MAX_EPOCHS = training_config["max_epochs"]
EARLY_STOPPING = training_config["early_stopping"]

torch.manual_seed(SEED)


class GCN(nn.Module):
    def __init__(
        self,
        n_hidden,
        n_features,
        n_classes,
        hidden_dim,
        dropout_rate,
        activation=F.relu,
        use_linear=False,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_linear = use_linear

        self.in_conv = GCNConv(n_features, self.hidden_dim)
        self.hidden_convs = nn.ModuleList(
            [
                GCNConv(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_hidden)
            ]
        )
        self.out_conv = GCNConv(self.hidden_dim, n_classes)
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.activation(self.in_conv(x, edge_index))

        for i in range(self.n_hidden):
            x = self.activation(self.hidden_convs[i](x, edge_index))
            if self.use_linear:
                x = Linear(self.hidden_dim, self.hidden_dim)

        x = self.activation(self.out_conv(x, edge_index))
        x = self.dropout(x)

        return x


def train_step(_model, _data, optim, _loss_fn):
    _model.train()
    optim.zero_grad()
    logits = _model(_data.x, _data.edge_index)[_data.train_mask]
    preds = logits.argmax(dim=1)
    y = _data.y[_data.train_mask]
    loss = _loss_fn(logits, y)
    acc = (preds == y).sum().item() / y.numel()
    loss.backward()
    optim.step()
    return loss.item(), acc


def eval_step(_model, _data, _loss_fn, stage):
    _model.eval()
    mask = getattr(_data, f"{stage}_mask")
    logits = _model(_data.x, _data.edge_index)[mask]
    preds = logits.argmax(dim=1)
    y = _data.y[mask]
    loss = _loss_fn(logits, y)
    acc = (preds == y).sum().item() / y.numel()
    return loss.item(), acc


def train(_model, _data, optim, _loss_fn, verbose=True):
    running_val_loss = []
    for epoch in range(MAX_EPOCHS):
        train_loss, train_acc = train_step(_model, _data, optim, _loss_fn)
        val_loss, val_acc = eval_step(_model, _data, _loss_fn, "val")
        running_val_loss.append(val_loss)

        if epoch > EARLY_STOPPING and val_loss < np.mean(
            running_val_loss[-(EARLY_STOPPING + 1) : -1]
        ):
            if verbose:
                print("Early stopping...")

            break

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    test_loss, test_acc = eval_step(_model, _data, _loss_fn, "test")
    print(f"Test loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    dataset = Planetoid(
        str(Path(data_config["data_dir"], data_config["cora_name"])),
        name="Cora",
        transform=T.NormalizeFeatures(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    model = GCN(
        1,
        num_features,
        num_classes,
        gcn_config["hidden_dim"],
        gcn_config["dropout"],
    )
    data = dataset[0]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=gcn_config["lr"],
        weight_decay=gcn_config["weight_decay"],
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    train(model, data, optimizer, loss_fn)