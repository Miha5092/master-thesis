import optuna
import argparse
import torch
import logging
import os
import numpy as np

from optuna.samplers import TPESampler
from typing import Callable
from torch.utils.data import DataLoader

from src.models.vitae import get_model, evaluate_loader
from src.utils.training import get_vitae_loss_fn, train
from src.datasets.vitae_dataset import load_data
from src.usage.vitae_training import main as full_training_pipeline


logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_suggestion_fn( 
    model_type: str,
    lamda1_values: list[float],
    lr_values: list[float],
    weight_decay_values: list[float],
    kernel_sizes: list[int],
):
    """
    Returns a function that suggests hyperparameters for a trial based on the defined search space.

    Args:
        model_type (str): Type of the model (e.g., 'tiny', 'lite', 'base', 'large', 'classic').
        lamda1_values (list[float]): List of lambda1 values to choose from.
        lr_values (list[float]): List of learning rate values to choose from.
        weight_decay_values (list[float]): List of weight decay values to choose from.
        kernel_sizes (list[int]): List of kernel sizes to choose from.

    Returns:
        Callable: A function that takes a trial object and suggests hyperparameters based on the defined search space.
    """

    def suggestion_fn(trial):
            lamda1 = trial.suggest_categorical("lamda1", lamda1_values)
            lr = trial.suggest_categorical("lr", lr_values)
            weight_decay = trial.suggest_categorical("weight_decay", weight_decay_values)

            # If using 'unet', allow kernel size tuning. Otherwise, use a default value.
            kernel_size = trial.suggest_categorical("kernel_size", kernel_sizes) if model_type == "unet" else 3

            return lamda1, lr, weight_decay, kernel_size
    return suggestion_fn


def objective(
    trial,
    model_type: str,
    sensor_type: str,
    sensor_number: int,
    batch_size: int,
    patch_size: tuple[int, int],
    suggestion_fn: Callable,
    epochs: int,
    seed: int = 42,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    early_stopping: bool = True,
    scaling_type: str = "min-max",
    timesteps: int = 1,
    split_mode: str = "monthly"
) -> float:
    """
    Objective function for hyperparameter optimization using Optuna.

    Args:
        trial (optuna.Trial): The trial object for hyperparameter optimization.
        model_type (str): Type of the model (e.g., 'tiny', 'lite', 'base', 'large', 'unet').
        sensor_type (str): Type of sensors used in the dataset.
        sensor_number (int): Number of sensors to use.
        batch_size (int): Batch size for training.
        patch_size (tuple[int, int]): Patch size for the model.
        suggestion_fn (Callable): Function to suggest hyperparameters.
        epochs (int): Number of epochs for training.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device to run the model on.
        early_stopping (bool): Whether to use early stopping during training.
        scaling_type (str): Type of scaling applied to the dataset.
        timesteps (int): Number of timesteps used in the dataset.
        split_mode (str): How to split the data ('monthly' or 'percentage').

    Returns:
        float: The mean relative error of the model on the validation set.
    """

    lamda1, lr, weight_decay, kernel_size = suggestion_fn(trial)

    # Load training and validation datasets.
    train_dataset, val_dataset, _, _ = load_data(
        val_size=0.1, test_size=0.1,
        sensor_type=sensor_type, 
        sensor_number=sensor_number, 
        split_mode=split_mode,
        combine_train_val=False,
        scaling_type=scaling_type,
        timesteps=timesteps,
        seed=seed
        )
    
    model = get_model(
        model_type=model_type,
        train_dataset=train_dataset,
        patch_size=tuple(patch_size),
        kernel_size=kernel_size
    )

    model, _, _ = train(
        experiment_name=f"Tuning_ViTAE",
        model=model,
        epochs=epochs,
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        lr=lr,
        weight_decay=weight_decay,
        evaluation_fn=evaluate_loader,
        loss_fn=get_vitae_loss_fn(lamda1),
        verbose=False,
        save=False,
        seed=seed,
        device=device,
        early_stopping=early_stopping,
    )

    _, relative_errors, *_ = evaluate_loader(
        model=model,
        loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        loss_fn=get_vitae_loss_fn(lamda1),
        device=device,
        advanced_statistics=True
    )

    return np.mean(relative_errors)


def hyperparameter_search(
    experiment_name: str,
    n_trials: int,
    model_type: str,
    sensor_type: str,
    sensor_number: int,
    batch_size: int,
    patch_size: tuple[int, int],
    lamda1_values: list[float],
    lr_values: list[float],
    weight_decay_values: list[float],
    kernel_sizes: list[int],
    epochs: int,
    seed: int = 42,
    train_final_model: bool = False,
    early_stopping: bool = True,
    scaling_type: str = "min-max",
    timesteps: int = 1,
    split_mode: str = "monthly"
):
    """
    Conducts a hyperparameter search for the specified model type using Optuna.

    Args:
        experiment_name (str): Name of the experiment for logging and saving results.
        n_trials (int): Number of trials to run for hyperparameter optimization.
        model_type (str): Type of the model to train (e.g., 'tiny', 'lite', 'base', 'large', 'unet').
        sensor_type (str): Type of sensors used in the dataset (e.g., 'fixed', 'random', 'real').
        sensor_number (int): Number of sensors used in the dataset.
        batch_size (int): Batch size for training.
        patch_size (tuple[int, int]): Patch size for the model.
        lamda1_values (list[float]): List of lambda1 values to choose from.
        lr_values (list[float]): List of learning rate values to choose from.
        weight_decay_values (list[float]): List of weight decay values to choose from.
        kernel_sizes (list[int]): List of kernel sizes to try.
        epochs (int): Number of epochs for training.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        train_final_model (bool, optional): Whether to train the final model after hyperparameter search. Defaults to False.
        early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
        scaling_type (str, optional): Type of scaling applied to the dataset. Defaults to "min-max".
        timesteps (int, optional): Number of timesteps used in the dataset. Defaults to 1.
        split_mode (str, optional): How to split the data ('monthly' or 'percentage'). Defaults to "monthly".

    Returns:
        None: The function saves the best hyperparameters and optionally trains the final model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the deterministic sampler.
    sampler = TPESampler(seed=seed)
    study_name = f"ViTAE-SL {model_type} with {scaling_type} scaling and {"real" if sensor_type == 'real' else f"{sensor_type}_{sensor_number}"} sensors"
    study = optuna.create_study(
        study_name=study_name, direction="minimize", sampler=sampler
    )

    suggestion_fn = get_suggestion_fn(
        model_type=model_type,
        lamda1_values=lamda1_values,
        lr_values=lr_values,
        weight_decay_values=weight_decay_values,
        kernel_sizes=kernel_sizes,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_type=model_type,
            sensor_type=sensor_type,
            sensor_number=sensor_number,
            batch_size=batch_size,
            patch_size=patch_size,
            suggestion_fn=suggestion_fn,
            epochs=epochs,
            seed=seed,
            device=device,
            early_stopping=early_stopping,
            scaling_type=scaling_type,
            timesteps=timesteps,
            split_mode=split_mode,
        ),
        n_trials=n_trials,
    )
    
    best_params = study.best_params
    print(f"Best params: {best_params}", flush=True)

    # Ensure the directory exists before saving.
    output_dir = "results/trained_models/vitae/params"
    os.makedirs(output_dir, exist_ok=True)
    params_path = os.path.join(output_dir, f"{experiment_name}_{model_type}_{"real" if sensor_type == "real" else f"{sensor_type}_{sensor_number}"}{"" if scaling_type == "min-max" else "_standard"}.pth")
    torch.save(best_params, params_path)

    if train_final_model:
        name = (
            f"{experiment_name}_"
            f"{model_type}_"
            f"{"real" if sensor_type == "real" else f"{sensor_type}_{sensor_number}"}"
            f"{"" if scaling_type == "min-max" else "_standard"}"
        )

        full_training_pipeline(
            experiment_name=name,
            model_type=model_type,
            sensor_type=sensor_type,
            sensor_number=sensor_number,
            patch_size=patch_size,
            batch_size=batch_size,
            lr=best_params.get("lr", 1e-4),
            weight_decay=best_params.get("weight_decay", 1e-5),
            epochs=500,
            lamda1=best_params.get("lamda1", 0.5),
            verbose=False,
            seed=seed,
            save_model=True,
            early_stopping=early_stopping,
            kernel_size=best_params.get("kernel_size", 3),
            scaling_type=scaling_type,
            timesteps=timesteps,
            split_mode="monthly"
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search the best hyperparameters for the ViTAE-SL model")
    parser.add_argument("--experiment_name", type=str, default="test", help="A name describing the experiment")
    parser.add_argument("-n", "--n_trials", type=int, default=20, help="Number of trials for hyperparameter search")
    parser.add_argument("--model_type", type=str, default="unet", help='Model type (e.g., "lite", "base", "large", or "unet")')
    parser.add_argument("--sensor_type", type=str, default="real", help="Type of sensors: fixed (30, 48, 108), fixed-random, random, or real")
    parser.add_argument("--sensor_number", type=int, default=30, help="Number of sensors")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--patch_size", nargs=2, type=int, default=[15, 10], help="Patch size (two integers)")
    parser.add_argument("--lamda1", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9], help="Lambda1 hyperparameter values")
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-4, 1e-3], help="Learning rate values")
    parser.add_argument("--weight_decay", nargs="+", type=float, default=[1e-5, 1e-4, 1e-3], help="Weight decay values")
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[3, 5, 7], help="Kernel sizes to try")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_final_model", action="store_true", help="Train final model with best parameters")
    parser.add_argument("-es", "--early_stopping", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--scaling_type", type=str, default="min-max", help='Type of scaling (e.g., "min-max", "standard")')
    parser.add_argument("--timesteps", type=int, default=1, help="How many consecutive timesteps to be used for a training example.")
    parser.add_argument("--split_mode", type=str, default="monthly", help="How to split the data. Between months or as percentages: 'monthly' or 'percentage'")

    args = parser.parse_args()

    hyperparameter_search(
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        model_type=args.model_type,
        sensor_type=args.sensor_type,
        sensor_number=args.sensor_number,
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size),
        lamda1_values=args.lamda1,
        lr_values=args.lr,
        weight_decay_values=args.weight_decay,
        kernel_sizes=args.kernel_sizes,
        epochs=args.epochs,
        seed=args.seed,
        train_final_model=args.train_final_model,
        early_stopping=args.early_stopping,
        scaling_type=args.scaling_type,
        timesteps=args.timesteps,
        split_mode=args.split_mode,
    )