import optuna
import argparse
import torch
import numpy as np
import os
import logging

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from typing import Callable

from src.models.vcnn import get_model, evaluate_loader
from src.datasets.voronoi_datasets import load_data, unscale
from src.utils.training import train, get_voronoi_loss_fn
from src.utils.evaluation import save_metrics_voronoi
from src.usage.vcnn_training import main as full_training_pipeline


def get_suggestion_fn( 
    model_type: str,
    lr_values: list[float],
    weight_decay_values: list[float],
    kernel_sizes: list[int],
    use_norm_list: list[bool],
    n_layers_list: list[int],
    hidden_channels_list: list[int],
) -> Callable:
    """
    Returns a function that suggests hyperparameters for a trial based on the defined search space.

    Args:
        model_type (str): Type of the model (e.g., 'tiny', 'lite', 'base', 'large', 'classic').
        lr_values (list[float]): List of learning rate values to choose from.
        weight_decay_values (list[float]): List of weight decay values to choose from.
        kernel_sizes (list[int]): List of kernel sizes to choose from.
        use_norm_list (list[bool]): List of boolean values indicating whether to use normalization.
        n_layers_list (list[int]): List of integers representing the number of layers for LSTM models.
        hidden_channels_list (list[int]): List of integers representing the number of hidden channels for ConvLSTM models.
    
    Returns:
        Callable: A function that takes a trial object and suggests hyperparameters based on the defined search space.
    """

    def suggestion_fn(trial):
        lr = trial.suggest_categorical("lr", lr_values)
        weight_decay = trial.suggest_categorical("weight_decay", weight_decay_values)

        kernel_size = trial.suggest_categorical("kernel_size", kernel_sizes) if model_type != "classic" else 3
        use_norm = trial.suggest_categorical("use_norm", use_norm_list) if model_type != "classic" else False

        n_layers = trial.suggest_categorical("n_layers", n_layers_list) if model_type == 'lstm' else 1
        hidden_channels = trial.suggest_categorical("hidden_channels", hidden_channels_list) if model_type == 'lstm' else 32

        return lr, weight_decay, kernel_size, use_norm, hidden_channels, n_layers
    return suggestion_fn


def objective(
    trial,
    model_type: str,
    sensor_type: str,
    sensor_number: int,
    batch_size: int,
    suggestion_fn: Callable,
    epochs: int,
    seed: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    early_stopping: bool = True,
    scaling_type: str = "min-max",
    timesteps: int = 1,
) -> float:
    """
    Objective function for hyperparameter optimization using Optuna.

    Args:
        trial (optuna.Trial): The trial object for hyperparameter optimization.
        model_type (str): Type of the model to train.
        sensor_type (str): Type of sensors used in the dataset.
        sensor_number (int): Number of sensors used in the dataset.
        batch_size (int): Batch size for training.
        suggestion_fn (Callable): Function to suggest hyperparameters for the trial.
        epochs (int): Number of epochs for training.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        device (torch.device, optional): Device to run the model on. Defaults to "cuda" if available, else "cpu".
        early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
        scaling_type (str, optional): Type of scaling applied to the dataset. Defaults to "min-max".
        timesteps (int, optional): Number of consecutive timesteps used for a training example. Defaults to 1.

    Returns:
        float: The mean relative error of the model on the validation set, which is minimized during optimization.
    """

    lr, weight_decay, kernel_size, use_norm, hidden_channels, n_layers = suggestion_fn(trial)

    # Load data with the chosen scaling type.
    train_dataset, val_dataset, _, _ = load_data(
        sensor_type=sensor_type, 
        sensor_number=sensor_number, 
        scaling_type=scaling_type,
        channel_timesteps=model_type != "lstm",  # If the model is ConvLSTM, we use channel_timesteps=False.
        combine_train_val=False,
        timesteps=timesteps,
        seed=seed
        )
    
    # Create the model based on the model type and parameters.
    model = get_model(
        model_type=model_type,
        train_dataset=train_dataset,
        kernel_size=kernel_size,
        use_norm=use_norm,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )

    # Train the model
    model, _, _ = train(
        experiment_name = f"Tuning_{model.__class__.__name__}",
        model=model,
        epochs=epochs,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        lr=lr,
        weight_decay=weight_decay,
        evaluation_fn=evaluate_loader,
        loss_fn=get_voronoi_loss_fn(),
        verbose=False,
        save=False,
        seed=seed,
        device=device,
        early_stopping=early_stopping,
    )

    # Evaluate the model on the validation set and return the relative error per validation set example.
    _, relative_errors, *_ = evaluate_loader(
        model=model,
        loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        loss_fn=get_voronoi_loss_fn(),
        device=device,
        advanced_statistics=True
    )

    return np.mean(relative_errors)  # Minimize the mean relative error


def search(
    experiment_name: str,
    n_trials: int,
    model_type: str,
    sensor_type: str,
    sensor_number: int,
    batch_size: int,
    lr_values: list[float],
    weight_decay_values: list[float],
    kernel_sizes: list[int],
    use_norm_list: list[bool],
    n_layers_list: list[int],
    hidden_channels_list: list[int],
    epochs: int,
    seed: int = 42,
    train_final_model: bool = False,
    early_stopping: bool = True,
    scaling_type: str = "min-max",
    timesteps: int = 1,
) -> None:
    """
    Conducts a hyperparameter search for the specified model type using Optuna.

    Args:
        experiment_name (str): Name of the experiment for logging and saving results.
        n_trials (int): Number of trials to run for hyperparameter optimization.
        model_type (str): Type of the model to train (e.g., 'tiny', 'lite', 'base', 'large', 'classic').
        sensor_type (str): Type of sensors used in the dataset (e.g., 'fixed', 'random', 'real').
        sensor_number (int): Number of sensors used in the dataset.
        batch_size (int): Batch size for training.
        lr_values (list[float]): List of learning rate values to choose from.
        weight_decay_values (list[float]): List of weight decay values to choose from.
        kernel_sizes (list[int]): List of kernel sizes to try.
        use_norm_list (list[bool]): List of boolean values indicating whether to use normalization.
        n_layers_list (list[int]): List of integers representing the number of layers for LSTM models.
        hidden_channels_list (list[int]): List of integers representing the number of hidden channels for ConvLSTM models.
        epochs (int): Number of epochs for training.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        train_final_model (bool, optional): Whether to train the final model with the best hyperparameters. Defaults to False.
        early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
        scaling_type (str, optional): Type of scaling applied to the dataset. Defaults to "min-max".
        timesteps (int, optional): Number of consecutive timesteps used for a training example. Defaults to 1.

    Returns:
        None: The function saves the best hyperparameters and optionally trains the final model.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up a deterministic sampler.
    sampler = TPESampler(seed=seed)
    study_name = f"VCNN {model_type} with {scaling_type} scaling and {"real" if sensor_type == 'real' else f"{sensor_type}_{sensor_number}"} sensors"
    study = optuna.create_study(
        study_name=study_name, direction="minimize", sampler=sampler
    )

    # Define the suggestion function based on the defined search space.
    suggestion_fn = get_suggestion_fn(
        model_type=model_type,
        lr_values=lr_values,
        weight_decay_values=weight_decay_values,
        kernel_sizes=kernel_sizes,
        use_norm_list=use_norm_list,
        n_layers_list=n_layers_list,
        hidden_channels_list=hidden_channels_list,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_type=model_type,
            sensor_type=sensor_type,
            sensor_number=sensor_number,
            batch_size=batch_size,
            suggestion_fn=suggestion_fn,
            epochs=epochs,
            seed=seed,
            device=device,
            early_stopping=early_stopping,
            scaling_type=scaling_type,
            timesteps=timesteps
        ),
        n_trials=n_trials,
    )
    
    best_params = study.best_params
    print(f"Best params: {best_params}", flush=True)

    # Ensure the directory exists before saving.
    output_dirs_map = {
        "VCNN": "results/trained_models/vunet",
        "VCNN_classic": "results/trained_models/vcnn",
        "ConvLSTM": "results/trained_models/clstm",
    }

    output_dir = output_dirs_map.get(model_type, "results/trained_models/unknown_model")
    os.makedirs(output_dir, exist_ok=True)
    params_path = os.path.join(output_dir, f"{experiment_name}_{"real" if sensor_type == "real" else f"{sensor_type}_{sensor_number}"}{"" if scaling_type == "min-max" else "_standard"}.pth")
    torch.save(best_params, params_path)

    # Optionally, train the final model with the best parameters.
    if train_final_model:
        name = (
            f"{experiment_name}_"
            f"{"real" if sensor_type == "real" else f"{sensor_type}_{sensor_number}"}"
            f"{"" if scaling_type == "min-max" else "_standard"}"
        )

        full_training_pipeline(
            experiment_name=name,
            model_type=model_type,
            seed=seed,
            sensor_type=sensor_type,
            sensor_number=sensor_number,
            scaling_type=scaling_type,
            timesteps=timesteps,
            batch_size=batch_size,
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            epochs=500,
            verbose=False,
            save_model=True,
            early_stopping=early_stopping,
            kernel_size=best_params.get("kernel_size", 3),
            use_norm=best_params.get("use_norm", False),
            hidden_channels=best_params.get("hidden_channels", 32),
            n_layers=best_params.get("n_layers", 1),
            split_mode="monthly"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search the best hyperparameters for the VCNN model")

    # --- General parameters ---
    parser.add_argument("--experiment_name", type=str, default="test", help="A name describing the experiment")
    parser.add_argument("--model_type", type=str, default="tiny", help="Model type (e.g., tiny, lite, base, large, classic)")
    parser.add_argument("-n", "--n_trials", type=int, default=20, help="Number of trials for hyperparameter search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--sensor_type", type=str, default="real", help="Type of sensors: fixed (30, 48, 108), fixed-random, random, or real")
    parser.add_argument("--sensor_number", type=int, default=30, help="Number of sensors")
    parser.add_argument("--scaling_type", type=str, default="min-max", help="Type of scaling (e.g., 'min-max', 'standard')")
    parser.add_argument("--timesteps", type=int, default=1, help="How many consecutive timesteps to be used for a training example.")

    # --- Training parameters ---
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train_final_model", action="store_true", default=False, help="Train the final model with the best hyperparameters")
    parser.add_argument("-es", "--early_stopping", action="store_true", default=False, help="Enable early stopping")

    # --- Searchable parameters ---
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-5, 1e-4], help="Learning rate values")
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3], help="Weight decay values")
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[3, 5, 7], help="Kernel sizes to try")
    parser.add_argument("--use_norm", nargs="+", type=bool, default=[True, False], help="Normalization usage options")
    parser.add_argument("--hidden_channels", nargs="+", type=int, default=[32, 64], help="Number of hidden channels for the ConvLSTM model.")
    parser.add_argument("--n_layers", nargs="+", type=int, default=[2, 3, 4], help="Number of LSTM layers for the ConvLSTM model.")

    args = parser.parse_args()

    search(
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        model_type=args.model_type,
        sensor_type=args.sensor_type,
        sensor_number=args.sensor_number,
        batch_size=args.batch_size,
        lr_values=args.lr,
        weight_decay_values=args.weight_decay,
        kernel_sizes=args.kernel_sizes,
        use_norm_list=args.use_norm,
        n_layers_list=args.n_layers,
        hidden_channels_list=args.hidden_channels,
        epochs=args.epochs,
        seed=args.seed,
        train_final_model=args.train_final_model,
        early_stopping=args.early_stopping,
        scaling_type=args.scaling_type,
        timesteps=args.timesteps
    )