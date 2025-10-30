# config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_layers: tuple = (64, 32)
    activation: str = 'relu'
    solver: str = 'adam'
    learning_rate: float = 0.01
    max_iter: int = 100
    early_stopping: bool = True
    validation_fraction: float = 0.2
    n_iter_no_change: int = 10
    alpha: float = 0.001
    random_state: int = 42

@dataclass
class DataConfig:
    test_size: float = 0.2
    random_state: int = 42
    data_path: str = "Modified_Churn_Modelling.csv"
