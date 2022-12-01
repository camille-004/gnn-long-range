import argparse
import sys
from pathlib import Path

import wandb
from src.models.train import prepare_training, train_module
from src.utils import load_config

data_config = load_config("data_config.yaml")
training_config = load_config("training_config.yaml")

parser = argparse.ArgumentParser()

# Set up CLI
parser.add_argument("classification_task", choices=["graph", "node"], type=str)
parser.add_argument("model", choices=["gat", "gcn", "gin", "gin_jk"], type=str)
parser.add_argument(
    "--max_epochs", default=training_config["max_epochs_default"], type=int
)
parser.add_argument("--activation", choices=["elu", "relu", "tanh"], type=str)
parser.add_argument("--n_hidden_layers", default=1, type=int)
parser.add_argument("--n_heads", default=1, type=int)
parser.add_argument("--jk_mode", default="none", type=str)
parser.add_argument("--plot_energy", action="store_true")
parser.add_argument("--plot_influence", action="store_true")

if __name__ == "__main__":
    test_names = Path(data_config["test_dir"], "test_name.txt")
    test_datasets = test_names.read_text().splitlines()[1:]

    args = parser.parse_args()
    wandb.login()

    task = args.classification_task

    if task == "graph" and any([args.plot_energy, args.plot_influence]):
        parser.error(
            "Plotting Dirichlet energy and neighborhood influence not"
            "supported for graph classification."
        )

    if task == "graph":
        dataset = test_datasets[0]
    else:
        dataset = test_datasets[1]

    data, model = prepare_training(
        task,
        args.model,
        args.n_hidden_layers,
        args.activation,
        dataset,
        num_heads=args.n_heads,
        mode=args.jk_mode,
    )

    if task == "graph":
        assert data.num_features == 136
        assert data.num_classes == 2
    else:
        assert data.num_features == 1433
        assert data.num_classes == 7

    results = train_module(
        data,
        model,
        max_epochs=args.max_epochs,
        plot_energies=args.plot_energy,
        plot_influence=args.plot_influence,
    )
    print(results)
    sys.exit()
