import numpy as np
import torch
import os

from src.datasets.real_obs_dataset import load_data as load_real_data

from src.datasets.vitae_dataset import load_data as load_vitae_data
from src.datasets.vitae_dataset import unscale as unscale_vitae

from src.datasets.voronoi_datasets import load_data as load_voronoi_data
from src.datasets.voronoi_datasets import unscale as unscale_voronoi

from src.utils.evaluation import compute_relative_error, compute_extra_metrics, compute_local_error


def add_extra_metrics(files: list[str], model_type: str, is_sparse: bool) -> None:
    print("-" * 50)

    if is_sparse:
        for i, file in enumerate(files):
            try:
                # Load existing arrays into a dictionary
                with np.load(file) as res:
                    data = {k: res[k] for k in res.keys()}

                preds = torch.from_numpy(data['predictions'])
                targets = torch.from_numpy(data['ground_truth'])

                dataset, _ = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=8, val_set=False)
                mask = torch.stack([target_mask for _, _, target_mask in dataset], dim=0)

                unscaled_l2_relative_errors = compute_relative_error(targets * mask, preds * mask)
                rmse, mfb, mfe = compute_extra_metrics(preds * mask, targets * mask, mask)

                print(f"File {i+1}/{len(files)} - RMSE: {round(rmse, 3)}, MFE: {round(mfe, 3)}, MFB: {round(mfb, 3)}")

                # Add new keys
                data.update({
                    "rmse": float(rmse),
                    "mfe": float(mfe),
                    "mfb": float(mfb),
                    "unscaled_errors": unscaled_l2_relative_errors
                })

                # Save everything back
                np.savez_compressed(file, **data)
            except Exception as e:
                print(f"Error processing file {file}")
                
                with open("text_log.text", "a") as f:
                    f.write(f"Error processing file {file}\n{e}\n")
    else:
        for i, file in enumerate(files):
            try:
                # Load existing arrays into a dictionary
                with np.load(file) as res:
                    data = {k: res[k] for k in res.keys()}

                preds = torch.from_numpy(data['predictions'])
                targets = torch.from_numpy(data['ground_truth'])

                unscaled_l2_relative_errors = compute_relative_error(targets, preds)
                rmse, mfb, mfe = compute_extra_metrics(preds, targets)

                print(f"File {i+1}/{len(files)} - RMSE: {round(rmse, 3)}, MFE: {round(mfe, 3)}, MFB: {round(mfb, 3)}")

                # Add new keys
                data.update({
                    "rmse": float(rmse),
                    "mfe": float(mfe),
                    "mfb": float(mfb),
                    "unscaled_errors": unscaled_l2_relative_errors
                })

                # Save everything back
                np.savez_compressed(file, **data)
            except Exception as e:
                print(f"Error processing file {file}")

                with open("text_log.text", "a") as f:
                    f.write(f"Error processing file {file}\n{e}\n")


def extra_metrics_main():
    add_extra_metrics([
        'results/predictions/vunet/random_random_5_predictions.npz',
        'results/predictions/vunet/random_random_10_predictions.npz',
        'results/predictions/vunet/random_random_15_predictions.npz',
        'results/predictions/vunet/random_random_20_predictions.npz',
        'results/predictions/vunet/random_random_25_predictions.npz',
        'results/predictions/vunet/random_random_30_predictions.npz',
    ], model_type='vunet', is_sparse=False)

    add_extra_metrics([
        'results/predictions/vitae/random_random_5_predictions.npz',
        'results/predictions/vitae/random_random_10_predictions.npz',
        'results/predictions/vitae/random_random_15_predictions.npz',
        'results/predictions/vitae/random_random_20_predictions.npz',
        'results/predictions/vitae/random_random_25_predictions.npz',
        'results/predictions/vitae/random_random_30_predictions.npz',
    ], model_type='vitae', is_sparse=False)

    add_extra_metrics([
        'results/predictions/vcnn/random_random_5_predictions.npz',
        'results/predictions/vcnn/random_random_10_predictions.npz',
        'results/predictions/vcnn/random_random_15_predictions.npz',
        'results/predictions/vcnn/random_random_20_predictions.npz',
        'results/predictions/vcnn/random_random_25_predictions.npz',
        'results/predictions/vcnn/random_random_30_predictions.npz',
    ], model_type="vcnn", is_sparse=False)

    add_extra_metrics([
        'results/predictions/kriging/random_random_5_predictions.npz',
        'results/predictions/kriging/random_random_10_predictions.npz',
        'results/predictions/kriging/random_random_15_predictions.npz',
        'results/predictions/kriging/random_random_20_predictions.npz',
        'results/predictions/kriging/random_random_25_predictions.npz',
        'results/predictions/kriging/random_random_30_predictions.npz',
    ], model_type='vitae', is_sparse=False)


def scale_back(files: list[str], model_type: str) -> None:
    print("-" * 50)

    for i, file in enumerate(files):
        try:
            # Load existing arrays into a dictionary
            with np.load(file) as res:
                data = {k: res[k] for k in res.keys()}

            old_preds = data['predictions']
            old_targets = data['ground_truth']
            old_local_errors = data["local_errors"]

            if model_type in ["vitae", "kriging"]:
                stats = load_vitae_data(split_mode="percent", sensor_type="random")[-1]

                if np.mean(old_preds) <= 2.0: 
                    preds = unscale_vitae(old_preds, scaling_type="min-max", **stats)
                else:
                    preds = old_preds
                
                if np.mean(old_targets) <= 2.0:
                    targets = unscale_vitae(old_targets, scaling_type="min-max", **stats)
                else:
                    targets = old_targets
            else:
                stats = load_voronoi_data(split_mode="percent", sensor_type="random")[-1]

                if np.mean(old_preds) <= 2.0:
                    preds = unscale_voronoi(old_preds, scaling_type="min-max", **stats)
                else:
                    preds = old_preds

                if np.mean(old_targets) <= 2.0:
                    targets = unscale_voronoi(old_targets, scaling_type="min-max", **stats)
                else:
                    targets = old_targets

            unscaled_local_error = compute_local_error(targets, preds)

            # Add new keys
            data.update({
                "original_predictions": old_preds,
                "original_ground_truth": old_targets,
                "original_local_error": old_local_errors,
                "predictions": preds,
                "ground_truth": targets,
                "local_errors": unscaled_local_error
            })

            # Save everything back
            np.savez_compressed(file, **data)

            print(f"File {i+1}/{len(files)} processed.")
        except Exception as e:
            print(f"Error processing file {file}")

            with open("text_log.text", "a") as f:
                f.write(f"Error processing file {file}\n{e}\n")


def scale_back_main():
    scale_back([
        'results/predictions/vunet/random_random_5_predictions.npz',
        'results/predictions/vunet/random_random_10_predictions.npz',
        'results/predictions/vunet/random_random_15_predictions.npz',
        'results/predictions/vunet/random_random_20_predictions.npz',
        'results/predictions/vunet/random_random_25_predictions.npz',
        'results/predictions/vunet/random_random_30_predictions.npz',
    ], model_type='vunet')

    scale_back([
        'results/predictions/vcnn/random_random_5_predictions.npz',
        'results/predictions/vcnn/random_random_10_predictions.npz',
        'results/predictions/vcnn/random_random_15_predictions.npz',
        'results/predictions/vcnn/random_random_20_predictions.npz',
        'results/predictions/vcnn/random_random_25_predictions.npz',
        'results/predictions/vcnn/random_random_30_predictions.npz',
    ], model_type='vcnn')

    scale_back([
        'results/predictions/kriging/random_random_5_predictions.npz',
        'results/predictions/kriging/random_random_10_predictions.npz',
        'results/predictions/kriging/random_random_15_predictions.npz',
        'results/predictions/kriging/random_random_20_predictions.npz',
        'results/predictions/kriging/random_random_25_predictions.npz',
        'results/predictions/kriging/random_random_30_predictions.npz',
    ], model_type='kriging')


if __name__ == "__main__":
    extra_metrics_main()