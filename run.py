import argparse
from pathlib import Path

import wandb
from src.models.train import prepare_training, train_module
from src.utils import load_config

config = load_config("data_config.yaml")

parser = argparse.ArgumentParser()

# Set up CLI
parser.add_argument("classification_task", choices=["graph", "node"], type=str)
parser.add_argument("model", choices=["gat", "gin"], type=str)
parser.add_argument("--n_hidden_layers", default=1, type=int)
parser.add_argument("--n_heads", default=1, type=int)

if __name__ == "__main__":
    test_names = Path(config["test_dir"], "test_name.txt")
    test_datasets = test_names.read_text().splitlines()[1:]

    args = parser.parse_args()
    wandb.login()

    task = args.classification_task

    if task == "graph":
        dataset = test_datasets[0]
    else:
        dataset = test_datasets[1]

    data, model = prepare_training(
        task, args.model, args.n_hidden_layers, dataset, num_heads=args.n_heads
    )

    if task == "graph":
        assert data.num_features == 136
        assert data.num_classes == 2
    else:
        assert data.num_features == 1433
        assert data.num_classes == 7

    results = train_module(data, model)
    print(results)
