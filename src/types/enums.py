from enum import Enum


class Activation(str, Enum):
    """Enum for activation function."""

    ELU = "elu"
    RELU = "relu"
    TANH = "tanh"


class Model(str, Enum):
    """Enum for model."""

    GAT = "gat"
    GCN = "gcn"
    GIN = "gin"
