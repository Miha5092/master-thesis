import torch
import numpy as np
import torch.nn as nn
import os
import logging

from torch.utils.data import DataLoader

from src.datasets.real_obs_dataset import load_data as load_real_data
from src.datasets.vitae_dataset import unscale
from src.utils.evaluation import compute_relative_error, compute_rmse, compute_mean_fractional_error, compute_mean_fractional_bias


def evaluate(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
    experiment_name: str = None,
) -> None:
    
    results_on_simulated = evaluate_on_simulated(model, data_scaling_type, timesteps)
    results_on_real = evaluate_on_real(model, data_scaling_type, timesteps)
    
    # Decide where to save the results
    save_dirs_map = {
        "VCNN": "results/predictions/vunet",
        "VCNN_classic": "results/predictions/vcnn",
        "ConvLSTM": "results/predictions/clstm",
        "OptimizedModule": "results/predictions/clstm",
        "ViTAE": "results/predictions/vitae"
    }

    # Decide where to save the results
    preds_dir = save_dirs_map.get(model.__class__.__name__, "results/predictions/unknown_model")
    os.makedirs(preds_dir, exist_ok=True)
    preds_file = os.path.join(preds_dir, f"{experiment_name}_results.npz")

    np.savez_compressed(
        preds_file,

        # ------ Saving the results on the simulated data ------

        # Save the data used to compute the metrics
        simulated_observations=results_on_simulated["observations"],
        simulated_ground_truths=results_on_simulated["ground_truths"],
        simulated_predictions=results_on_simulated["predictions"],
        simulated_target_masks=results_on_simulated["target_masks"],
        # Save the relative error
        simulated_global_re=results_on_simulated["global_re"],
        simulated_pollutants_re=results_on_simulated["pollutants_re"],
        # Save the RMSe
        simulated_global_rmse=results_on_simulated["global_rmse"],
        simulated_pollutants_rmse=results_on_simulated["pollutants_rmse"],
        # Save the MFE
        simulated_global_mfe=results_on_simulated["global_mfe"],
        simulated_pollutants_mfe=results_on_simulated["pollutants_mfe"],
        # Save the MFB
        simulated_global_mfb=results_on_simulated["global_mfb"],
        simulated_pollutants_mfb=results_on_simulated["pollutants_mfb"],


        # ------ Saving the results on the real data ------

        # Save the data used to compute the metrics
        real_observations=results_on_real["observations"],
        real_ground_truths=results_on_real["ground_truths"],
        real_predictions=results_on_real["predictions"],
        real_target_masks=results_on_real["target_masks"],
        # Save the relative error
        real_global_re=results_on_real["global_re"],
        real_pollutants_re=results_on_real["pollutants_re"],
        # Save the RMSe
        real_global_rmse=results_on_real["global_rmse"],
        real_pollutants_rmse=results_on_real["pollutants_rmse"],
        # Save the MFE
        real_global_mfe=results_on_real["global_mfe"],
        real_pollutants_mfe=results_on_real["pollutants_mfe"],
        # Save the MFB
        real_global_mfb=results_on_real["global_mfb"],
        real_pollutants_mfb=results_on_real["pollutants_mfb"]
    )


def evaluate_on_simulated(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
) -> dict:
    pass


def evaluate_on_real(
    model: nn.Module,
    data_scaling_type: str,
    timesteps: int,
) -> dict:
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    model_names_map = {
        "VCNN": "vunet",
        "VCNN_classic": "vcnn",
        "ConvLSTM": "clstm",
        "OptimizedModule": "clstm",
        "ViTAE": "vitae"
    }
    model_type = model_names_map.get(model.__class__.__name__)
    
    dataset, stats = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=timesteps, val_set=False)
    dataloader = DataLoader(
        dataset,
        batch_size=64 if model_type != "clstm" else 32,
        shuffle=False, num_workers=3, pin_memory=True
    )

    model.to(device)

    observations, ground_truths, predictions, target_masks = [], [], [], []

    for obs, ground_truth, target_mask in dataloader:
        obs = obs.to(device).float()

        preds = model(obs)

        # If we are working with the ConvLSTM model, we need to take the last timestep
        if model_type == "clstm":
            preds = preds[:, -1]

        # The ViTAE model outputs both the encoder and decoder predictions, we only need the decoder predictions
        if model_type == "vitae":
            preds = preds[-1]

        # Scale the data back to its original values
        channel_count = stats["data_min"].shape[1]
        obs_shape = obs.shape

        obs = unscale(obs.reshape(-1, channel_count, obs.shape[-2], obs.shape[-1]).cpu().detach().numpy(), data_scaling_type, **stats).reshape(*obs_shape)
        ground_truth = unscale(ground_truth.float().numpy(), data_scaling_type, **stats)
        target_mask = target_mask.float().numpy()
        preds = unscale(preds.cpu().detach().numpy(), data_scaling_type, **stats)

        # Store the observations, ground truth, target masks, and predictions
        observations.append(obs)
        ground_truths.append(ground_truth)
        target_masks.append(target_mask)
        predictions.append(preds)

    observations = np.concatenate(observations)
    ground_truths = torch.from_numpy(np.concatenate(ground_truths))
    target_masks = torch.from_numpy(np.concatenate(target_masks))
    predictions = torch.from_numpy(np.concatenate(predictions))

    # ------- Compute the metrics -------

    # These metrics are computed globally (all pollutants together)
    global_re = np.mean(compute_relative_error(ground_truths * target_masks, predictions * target_masks))
    global_rmse = compute_rmse(ground_truths, predictions, target_masks).item()
    global_mfe = compute_mean_fractional_error(ground_truths, predictions, target_masks).item()
    global_mfb = compute_mean_fractional_bias(ground_truths, predictions, target_masks).item()

    # These metrics are computed for each pollutant separately
    pollutants_re, pollutants_rmse, pollutants_mfe, pollutants_mfb = [], [], [], []
    for i in range(4):
        pollutant_ground_truths = ground_truths[:, i]
        pollutant_predictions = predictions[:, i]
        pollutant_target_masks = target_masks[:, i]

        pollutant_re = np.mean(compute_relative_error(pollutant_ground_truths * pollutant_target_masks, pollutant_predictions * pollutant_target_masks))    
        pollutant_rmse = compute_rmse(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks).item()
        pollutant_mfe = compute_mean_fractional_error(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks).item()
        pollutant_mfb = compute_mean_fractional_bias(pollutant_ground_truths, pollutant_predictions, pollutant_target_masks).item()

        pollutants_re.append(pollutant_re)
        pollutants_rmse.append(pollutant_rmse)
        pollutants_mfe.append(pollutant_mfe)
        pollutants_mfb.append(pollutant_mfb)

    return {
        # Save the data used to compute the metrics
        "observations": observations,
        "ground_truths": ground_truths.numpy(),
        "predictions": predictions.numpy(),
        "target_masks": target_masks.numpy(),
        # Save the relative error
        "global_re": global_re,
        "pollutants_re": pollutants_re,
        # Save the RMSe
        "global_rmse": global_rmse,
        "pollutants_rmse": pollutants_rmse,
        # Save the MFE
        "global_mfe": global_mfe,
        "pollutants_mfe": pollutants_mfe,
        # Save the MFB
        "global_mfb": global_mfb,
        "pollutants_mfb": pollutants_mfb
    }