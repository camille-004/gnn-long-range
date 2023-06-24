import argparse
import sys

import wandb
from src.models.train import prepare_training, train_module
from src.utils import load_config, get_group_name

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_config = load_config("data_config.yaml")
training_config = load_config("training_config.yaml")

parser = argparse.ArgumentParser()

# Set up CLI
parser.add_argument(
    "model",
    choices=["gat", "gcn", "gin", "gin_jk", "sognn"],
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
    "-t",
    "--add_edges_thres",
    default=0.0,
    type=float,
    help="Threshold, as a percentage of original edge cardinality, "
    "for amount of new random edges to add",
)
parser.add_argument(
    "--n_heads",
    default=1,
    type=int,
    help="Number of heads for multi-head attention. GATs only!",
)
parser.add_argument(
    "-r",
    "--distance",
    default=5,
    type=int,
    help="根据邻接矩阵的r次方选择远的邻接矩阵",
)
parser.add_argument(
    "--plot_energy",
    action="store_true",
    help="Plot Dirichlet energy of each layer.",
)
parser.add_argument(
    "--plot_rayleigh",
    action="store_true",
    help="Plot Rayleigh quotient of each layer.",
)
parser.add_argument(
    "--plot_influence",
    action="store_true",
    help="Plot up to r-th-order neighborhood influence on a random node.",
)
parser.add_argument(
    "-pn",
    "--project_name",
    default='sognn_no_name',
    type=str,
    help='name of the project on wandb website',
)
parser.add_argument(
    "--fa",
    action="store_true",
    help="Add a fully connected layer."
)
parser.add_argument(
    "--tud",
    action='store_true',
    help='Make the graph to undirected.'
)
parser.add_argument(
    "--unl",
    action='store_true',
    help='Use sognn loss.'
)
parser.add_argument(
    "-o",
    "--ordered",
    default=True,
    type=bool,
    help="Either to use the ordered scheme."
)
if __name__ == "__main__":
    args = parser.parse_args()

    dataset_name=args.dataset

    if dataset_name in ['cora', 'pubmed', 'citeseer']:
        wandb.init(project=args.project_name)
        data, model = prepare_training(
            args.model,
            args.n_hidden_layers,
            args.add_edges_thres,
            args.activation,
            dataset_name=dataset_name,
            n_heads=args.n_heads,
            r=args.distance,
            fa=args.fa,
            tud=args.tud,
            unl=args.unl,
            ordered=args.ordered,
        )

        results = train_module(
            data,
            model,
            max_epochs=args.max_epochs,
            plot_energies=args.plot_energy,
            plot_rayleigh=args.plot_rayleigh,
            plot_influence=args.plot_influence,
            r=args.distance,
        )
    
    if dataset_name in ['Actor', 'texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel']:
        group_name = get_group_name(args)
        acc = 0
        for split in range(10):
            wandb.init(
                project=args.project_name, 
                group=group_name, 
                config=args, 
                name=f'{dataset_name}_split_{split}_{group_name}'
                )
            data, model = prepare_training(
                args.model,
                args.n_hidden_layers,
                args.add_edges_thres,
                args.activation,
                dataset_name=dataset_name,
                n_heads=args.n_heads,
                r=args.distance,
                fa=args.fa,
                tud=args.tud,
                unl=args.unl,
                split=split
            )

            results = train_module(
                data,
                model,
                max_epochs=args.max_epochs,
                plot_energies=args.plot_energy,
                plot_rayleigh=args.plot_rayleigh,
                plot_influence=args.plot_influence,
                r=args.distance,
            )
            acc_ = results['node_SOGNN']['test_accuracy']
            acc += 0.1 * acc_
        print('=' * 30)
        print(f'Average accuracy on 10 splits: {acc:.3f}.')
        print('=' * 30)
        wandb.init(project=args.project_name, group=group_name, config=args, name=group_name+'_Acc')
        wandb.log({'Average accuracy': acc})
        wandb.finish()

    sys.exit()
