import argparse
import logging
import torch

from src.usage.vcnn_training import main as vcnn_main
from src.usage.vitae_training import main as vitae_main

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(
    model_type: str,
    tested_time_jumps: list[int],
    seed: int = 42,
):
    """
    Run experiments with varying timesteps.

    Args:
        model_type (str): Type of model to train. Options are 'vcnn', 'vunet', 'vitae', or 'clstm'.
        tested_time_jumps (list[int]): List of hours to jump between sequence examples.
        seed (int): Random seed for reproducibility. Default is 42.
        
    Raises:
        ValueError: If an unknown model type is provided.
    """

    for time_jump in tested_time_jumps:
        experiment_name = f"delay_{time_jump}_t8"
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
                timesteps=8,
                timesteps_jump=time_jump
            )
        elif model_type == "vcnn":
            logging.error(f"The best performing VCNN model does not use multiple timesteps so you cannot use it for this experiment.")
            return
        elif model_type in ["vunet", "clstm"]:
            training_model_type = "lstm" if model_type == "clstm" else "tiny"
            batch_size = 16 if model_type == "clstm" else 64

            vcnn_main(
                experiment_name=experiment_name,
                model_type=training_model_type,
                seed=seed,
                sensor_type="real",
                sensor_number=30,
                scaling_type="min-max",
                timesteps=8,
                timesteps_jump=time_jump,
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
    parser.add_argument("--tested_time_jumps", type=int, nargs='+', default=[2, 3, 4], help="List of hours to jump between sequence examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        tested_time_jumps=args.tested_time_jumps,
        seed=args.seed,
    )