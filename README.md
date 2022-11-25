# gnn-long-range
Repository for Exploring Graph Neural Networks for Long Range Interactions. I am currently only working on testing baseline SotA GNN architectures.

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
