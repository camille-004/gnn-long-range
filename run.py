import argparse
import sys

import wandb
from src.models.train import prepare_training, train_module
from src.utils import load_config

data_config = load_config("data_config.yaml")
training_config = load_config("training_config.yaml")

parser = argparse.ArgumentParser()

# Set up CLI
parser.add_argument(
    "classification_task",
    choices=["graph", "node"],
    type=str,
    help="Classification task: either graph or node.",
)
parser.add_argument(
    "model",
    choices=["gat", "gcn", "gin", "gin_jk"],
    type=str,
    help="Name of chosen model. gin_jk only supported by the graph "
    "classification task.",
)
parser.add_argument(
    "-e",
    "--max_epochs",
    default=training_config["max_epochs_default"],
    type=int,
    help="Maximum number of epochs to run model, if early stopping not "
    "converged.",
)
parser.add_argument(
    "-d",
    "--dataset",
    choices=data_config["test_datasets"],
    default=data_config["node"]["node_data_name_default"],
    type=str,
    help="Name of dataset on which to train model.",
)
parser.add_argument(
    "-a",
    "--activation",
    choices=["elu", "relu", "tanh"],
    type=str,
    help="Activation function used by neural network.",
)
parser.add_argument(
    "-nh",
    "--n_hidden_layers",
    default=1,
    type=int,
    help="Number of hidden layers to include in neural network.",
)
parser.add_argument(
    "--n_heads",
    default=1,
    type=int,
    help="Number of heads for multi-head attention. GATs only!",
)
parser.add_argument(
    "--jk_mode",
    default="none",
    type=str,
    help="Mode of jumping knowledge for graph classifaction gin_jk.",
)
parser.add_argument(
    "--plot_energy",
    action="store_true",
    help="Plot Dirichlet energy pf each layer.",
)
parser.add_argument(
    "--plot_influence",
    action="store_true",
    help="Plot up to r-th-order neighborhood influence on a random node.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.login()

    task = args.classification_task

    if args.model == "gin_jk":
        assert (
            task == "graph"
        ), "gin_jk only supported by graph classification."

    if task == "graph" and any([args.plot_energy, args.plot_influence]):
        parser.error(
            "Plotting Dirichlet energy and neighborhood influence not"
            "supported for graph classification."
        )

    data, model = prepare_training(
        task,
        args.model,
        args.n_hidden_layers,
        args.activation,
        dataset_name=args.dataset,
        num_heads=args.n_heads,
        mode=args.jk_mode,
    )

    results = train_module(
        data,
        model,
        max_epochs=args.max_epochs,
        plot_energies=args.plot_energy,
        plot_influence=args.plot_influence,
    )

    sys.exit()
