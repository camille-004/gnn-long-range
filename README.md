# gnn-long-range
This project contains customizable baseline SoTA message passing graph neural network (GNN) architectures. Training functionality is provided by `pytorch_lightning`, and logging by [Weights & Biases](https://wandb.ai/site). It also consists of experiments attempting to visualize assessments of **oversmoothing** (Rayleigh constant, Dirichlet energy) and **oversquashing** (embedding Jacobian) for node classification. The main purpose of this repository is to explore the limitations of MPNNs for long-range interactions.

## Project Structure

```shell
├── config  # Contains all config files
├── logs  # Contains results.csv from experiments
├── notebooks  # Notebooks, to be used for presentation
├── references  # List of papers and codebases referenced
├── reports  # Reports and figures generated by run script
├── scripts  # Run shell scripts for experiments
└── src
    └── models
        ├── graph_classification  # All architectures for graph classification.
        ├── node_classification  # All architectures for node classification.
        └── train.py  # Training script, based on PyTorch Lightning's `Trainer`.
    ├── data_modules.py  # Definitions of graph and node `LightningDataModule`s.
    ├── utils.py  # Utility functions (load configuration, computations).
    └── visualize.py  # Model graphs to save to reports/figures.

```

## Prerequisites

This project is built on `conda` and Python 3.10.

**GPU tools:** These models are built using `torch` v1.13.0 and CUDA v11.7, and this is reflected in `environment_gpu.yaml`. You may change your CUDA and `torch` versions in `environment_gpu.yaml`.

To install all necessary dependencies, create a new `conda` environment:
```shell
conda env create -f environment_cpu.yaml  # CPU environment
conda env create -f environment_gpu.yaml  # GPU environment
```

In case the `pip` installations hang when running the above, run the following after all conda dependencies are installed.
```shell
pip install -r requirements_cpu.txt  # For CPU environment
pip install -r requirements_gpu.txt  # For GPU environment
```

## Usage

To run the experiments from the report, simply execute `scripts/run.sh`. The script starts by emptying `logs/results.csv` and `reports/figures`, in which oversmoothing and oversquashing plots will be stored.

Alternatively, to run your own, execute `run.py` with the following parameters:
- `classification_task` - Classification task: either graph or node.
- `model` - Name of chosen model. gin_jk only supported by the graph classification task.
- `-e, --max_epochs` - *optional*, Maximum number of epochs to run model, if early stopping not converged.
- `-d, --dataset` - *optional*, Name of dataset on which to train model.
- `-a, --activation` - *optional*, Activation function used by neural network.
- `-nh, --n_hidden_layers` - *optional*, Number of hidden layers to include in neural network.
- `--n_heads` - *optional*, Number of heads for multi-head attention. GATs only!
- `--jk_mode` - *optional*, Mode of jumping knowledge for graph classifaction gin_jk.
- `--plot_energy` - *optional*, Plot Dirichlet energy pf each layer.
- `--plot_influence` - *optional*, Plot up to r-th-order neighborhood influence on a random node.

You may edit any model hyperparameters, or data and training parameters in the files in the `config` directory.

**Note**: If you get an empty Jacobian when getting the influence scores (resulting in a `seaborn` error), you most likely randomly chose an isolated node. For now, a simple fix would be to change the `seed` in `global_config`.

## Results
