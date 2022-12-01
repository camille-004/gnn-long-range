# gnn-long-range
This project contains customizable baseline SoTA graph neural network (GNN) architectures. Training functionality is provided by `pytorch_lightning`, and logging by [Weights & Biases](https://wandb.ai/site).

## Project Structure
- `config` - Config files for training, model building, and data loading/preprocessing.
- `notebooks` - To be used more in the future for presentation.
- `src` - Source code for this project.
    - `data` - Data loading and preprocessing script(s).
        - `data_modules.py` - `LightningDataModule`s for graph and node classification datasets, loaded from `torch_geometric`.
    - `models` - Baseline model architecture definitions.
        - `graph_classification` - Model definitions for the graph classification task.
            - `base.py` - Definition for the `BaseGraphClassifier`.
            - The rest of the scripts define the architectures we are testing out.
        - `node_classification` - Model definitions for the node classification task.
            - `base.py` - Definition for the `BaseNodeClassifier`.
            - The rest of the scripts define the architectures we are testing out.
        - `train.py` - Training script, based on PyTorch Lightning's `Trainer`.
    - `visualization` - Will soon house scripts for generating explanatory visualizations (embedding predictions, etc.).
    - `utils.py` - Utility functions, including loading a configuration.

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

To run the experiments from the report, simply execute `run.sh`. Alternatively, to run your own:
```shell
python run.py \
--
```

You may edit any model hyperparameters, or data and training parameters in the files in the `config` directory.
