import argparse
import logging
import torch

from src.usage.vcnn_training import main as vcnn_main
from src.usage.vitae_training import main as vitae_main

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(
    model_type: str,
    sensor_numbers: list[int],
    seed: int = 42,
):
    """
    Run experiments with random sensor configurations.

    Args:
        model_type (str): Type of model to train. Options are 'vcnn', 'vunet', 'vitae', or 'clstm'.
        sensor_numbers (list[int]): List of sensor numbers to test.
        seed (int): Random seed for reproducibility. Default is 42.
    Raises:
        ValueError: If an unknown model type is provided.
    """

    for sensor_number in sensor_numbers:
        experiment_name = f"random_{sensor_number}"
        best_params = torch.load(f"results/trained_models/{model_type}/params/random_30.pth")

        if model_type == "vitae":
            vitae_main(
                experiment_name=experiment_name,
                model_type="unet",
                sensor_type="random",
                sensor_number=sensor_number,
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
                timesteps=1,
            )
        elif model_type in ["vcnn", "vunet", "clstm"]:

            if model_type == "vcnn":
                training_model_type = "classic"
            elif model_type == "vunet":
                training_model_type = "tiny"
            else:
                training_model_type = "lstm"

            batch_size = 32 if model_type == 'lstm' else 64

            vcnn_main(
                experiment_name=experiment_name,
                model_type=training_model_type,
                seed=seed,
                sensor_type="random",
                sensor_number=sensor_number,
                scaling_type="min-max",
                timesteps=500,
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
    parser.add_argument("--sensor_number", type=int, nargs='+', default=[5, 10, 15, 20, 25, 30], help="List of sensor numbers to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        sensor_numbers=args.sensor_number,
        seed=args.seed,
    )