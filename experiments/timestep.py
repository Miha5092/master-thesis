import argparse
import logging
import torch

from src.usage.vcnn_training import main as vcnn_main
from src.usage.vitae_training import main as vitae_main

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_batch_size(model_type: str, timesteps: int) -> int:
    """
    Method built to dynamically change the batch size for ConvLSTM models. This was
    done because the model has a very large memory footprint.

    All other models retain a batch size of 64
    """

    if model_type != "clstm":
        return 64
    
    if timesteps < 8:
        return 32
    
    return 16


def main(
    model_type: str,
    tested_timesteps: list[int],
    seed: int = 42,
):
    """
    Run experiments with varying timesteps.

    Args:
        model_type (str): Type of model to train. Options are 'vcnn', 'vunet', 'vitae', or 'clstm'.
        tested_timesteps (list[int]): List of timesteps to test.
        seed (int): Random seed for reproducibility. Default is 42.
        
    Raises:
        ValueError: If an unknown model type is provided.
    """

    for timesteps in tested_timesteps:
        experiment_name = f"t{timesteps}"
        best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")

        if model_type == "vitae":
            vitae_main(
                experiment_name=experiment_name,
                model_type="unet",
                sensor_type="real",
                sensor_number=30,
                patch_size=(15, 10),
                batch_size=64,
                lr=best_params.get("lr", 1e-4),
                weight_decay=best_params.get("weight_decay", 1e-5),
                epochs=500,
                lamda1=best_params.get("lamda1", 0.01),
                verbose=False,
                seed=seed,
                save_model=True,
                early_stopping=True,
                kernel_size=best_params.get("kernel_size", 3),
                scaling_type="min-max",
                timesteps=timesteps,
            )
        elif model_type in ["vcnn", "vunet", "clstm"]:

            if model_type == "vcnn":
                training_model_type = "classic"
            elif model_type == "vunet":
                training_model_type = "tiny"
            else:
                training_model_type = "lstm"

            batch_size = get_batch_size(model_type, timesteps)

            vcnn_main(
                experiment_name=experiment_name,
                model_type=training_model_type,
                seed=seed,
                sensor_type="real",
                sensor_number=30,
                scaling_type="min-max",
                timesteps=timesteps,
                batch_size=batch_size,
                lr=best_params.get("lr", 1e-4),
                weight_decay=best_params.get("weight_decay", 1e-5),
                epochs=500,
                verbose=False,
                save_model=True,
                early_stopping=True,
                kernel_size=best_params.get("kernel_size", 3),
                use_norm=best_params.get("use_norm", True),
                hidden_channels= best_params.get("hidden_channels", 64),
                n_layers= best_params.get("n_layers", 3),
            )
        else:
            logging.error(f"Unknown model type: {model_type}. Please choose from 'vcnn', 'vunet', 'vitae', or 'clstm'.")
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with specified timesteps")
    parser.add_argument("--model_type", type=str, help="Model type: 'vcnn', 'vunet', 'vitae', or 'clstm'")
    parser.add_argument("--tested_timesteps", type=int, nargs='+', default=[1, 2, 3, 4, 6, 8, 12], help="List of timesteps to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        tested_timesteps=args.tested_timesteps,
        seed=args.seed,
    )