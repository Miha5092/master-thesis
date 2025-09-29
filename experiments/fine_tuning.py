import argparse
import logging
import torch

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from torch.optim import AdamW
from itertools import zip_longest

from src.utils.evaluation import evaluate_on_real
from src.usage.vcnn_training import main as voronoi_main
from src.usage.vitae_training import main as vitae_main
from src.datasets.real_obs_dataset import load_data as load_real_data
from src.utils.training import get_scheduler, get_voronoi_loss_fn, safe_scheduler_step, get_vitae_loss_fn
from src.datasets.voronoi_datasets import load_data
from src.datasets.vitae_dataset import load_data as load_vitae_data

logging.basicConfig(level=logging.INFO, format="%(message)s")


def fine_tune_vunet(
    experiment_name: str,
    epochs: int = 500,
    fine_tuning_epochs: int = 100,
    timesteps: int = 8,
    batch_size: int = 64,
    noise: str = 'none',
    full_noise: bool = True,
    seed: int = 42,
) -> torch.nn.Module:
    model_type = "vunet"

    best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")

    model = voronoi_main(
        experiment_name=experiment_name,
        model_type="tiny",
        seed=seed,
        sensor_type="real-random",
        sensor_number=30,
        scaling_type="min-max",
        timesteps=timesteps,
        timesteps_jump=1,
        batch_size=batch_size,
        lr=best_params.get("lr", 1e-4),
        weight_decay=best_params.get("weight_decay", 1e-5),
        epochs=0,
        verbose=False,
        save_model=False,
        early_stopping=False,
        kernel_size=best_params.get("kernel_size", 3),
        use_norm=best_params.get("use_norm", True),
        hidden_channels=best_params.get("hidden_channels", 64),
        n_layers=best_params.get("n_layers", 3),
        noise=noise,
        full_noise=full_noise,
    )

    train_dataset, *_ = load_data(
        sensor_type="real-random",
        sensor_number=30,
        split_mode="monthly",
        scaling_type="min-max",
        combine_train_val=True,
        timesteps=timesteps,
        timesteps_jump=1,
        channel_timesteps=True,
        noise=noise,
        full_noise=full_noise,
        seed=seed
        )
    fine_tuning_dataset, _ = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=timesteps, val_set=False, test_set=False)
    
    dense_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
    sparse_loader = DataLoader(fine_tuning_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)

    seed_everything(seed=seed, verbose=False)

    # Initializing training utils
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=best_params.get("lr", 1e-4), weight_decay=best_params.get("weight_decay", 1e-5))
    scheduler = get_scheduler(optimizer, fine_tuning_epochs)
    loss_fn = get_voronoi_loss_fn()

    for _ in range(fine_tuning_epochs):
        model.train()

        dense_iter = iter(dense_loader)
        sparse_iter = iter(sparse_loader)

        for batch_dense, batch_sparse in zip_longest(dense_iter, sparse_iter):
            if batch_dense is None or batch_sparse is None:
                continue
            
            obs_sparse, target_sparse, sparse_target_mask = batch_sparse
            obs_sparse, target_sparse, sparse_target_mask = obs_sparse.to(device).float(), target_sparse.to(device).float(), sparse_target_mask.to(device).float()

            # Train the model to reconstruct the sparse observations

            pred_sparse = model(obs_sparse)
            loss_sparse = loss_fn(pred_sparse * sparse_target_mask, target_sparse * sparse_target_mask)

            optimizer.zero_grad()
            loss_sparse.backward()
            optimizer.step()

            # Correct its predictions based on the dense observations

            obs_dense, target_dense = batch_dense
            obs_dense, target_dense = obs_dense.to(device).float(), target_dense.to(device).float()

            torch.clamp(obs_dense, 0, 1, out=obs_dense)

            pred_dense = model(obs_dense)
            loss_dense = loss_fn(pred_dense, target_dense)

            optimizer.zero_grad()
            loss_dense.backward()
            optimizer.step()
        
        safe_scheduler_step(scheduler)

    return model

def fine_tune_vitae(
    experiment_name: str,
    epochs: int = 500,
    fine_tuning_epochs: int = 100,
    timesteps: int = 8,
    batch_size: int = 64,
    noise: str = 'none',
    full_noise: bool = True,
    seed: int = 42,
) -> torch.nn.Module:
    model_type = "vitae"

    best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")
    model = vitae_main(
        experiment_name=experiment_name,
        model_type="unet",
        sensor_type="real-random",
        sensor_number=30,
        patch_size=(15, 10),
        batch_size=32,
        lr=best_params.get("lr", 1e-4),
        weight_decay=best_params.get("weight_decay", 1e-5),
        epochs=0,
        lamda1=best_params.get("lamda1", 0.5),
        verbose=False,
        seed=seed,
        save_model=True,
        early_stopping=False,
        kernel_size=best_params.get("kernel_size", 3),
        scaling_type="min-max",
        timesteps=timesteps,
        timesteps_jump=1,
        noise=noise,
        full_noise=full_noise,
    )

    train_dataset, *_ = load_vitae_data(
        combine_train_val=True,
        sensor_type="real-random",
        timesteps=timesteps,
        noise=noise,
        full_noise=full_noise,
        seed=seed
    )

    fine_tuning_dataset, _ = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=timesteps, val_set=False, test_set=False)
    
    dense_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
    sparse_loader = DataLoader(fine_tuning_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)

    seed_everything(seed=seed, verbose=False)

    # Initializing training utils
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=best_params.get("lr", 1e-4), weight_decay=best_params.get("weight_decay", 1e-5))
    scheduler = get_scheduler(optimizer, fine_tuning_epochs)
    loss_fn = get_vitae_loss_fn(best_params.get("lamda1", 0.5))

    for _ in range(fine_tuning_epochs):
        model.train()

        dense_iter = iter(dense_loader)
        sparse_iter = iter(sparse_loader)

        for batch_dense, batch_sparse in zip_longest(dense_iter, sparse_iter):
            if batch_dense is None or batch_sparse is None:
                continue
            
            obs_sparse, target_sparse, sparse_target_mask = batch_sparse
            obs_sparse, target_sparse, sparse_target_mask = obs_sparse.to(device).float(), target_sparse.to(device).float(), sparse_target_mask.to(device).float()

            # Train the model to reconstruct the sparse observations

            pred_sparse_enc, pred_sparse_dec = model(obs_sparse)

            loss_sparse = loss_fn(
                pred_sparse_enc * sparse_target_mask, 
                pred_sparse_dec * sparse_target_mask,
                sparse_target_mask,
                target_sparse * sparse_target_mask
            )

            optimizer.zero_grad()
            loss_sparse.backward()
            optimizer.step()

            # Correct its predictions based on the dense observations

            obs_dense, target_dense, mask = batch_dense
            obs_dense, target_dense = obs_dense.to(device).float(), target_dense.to(device).float()

            torch.clamp(obs_dense, 0, 1, out=obs_dense)

            pred_dense_enc, pred_dense_dec = model(obs_dense)
            loss_dense = loss_fn(
                pred_dense_enc, 
                pred_dense_dec,
                mask,
                target_dense
            )

            optimizer.zero_grad()
            loss_dense.backward()
            optimizer.step()
        
        safe_scheduler_step(scheduler)

    return model

def fine_tune_clstm(
    experiment_name: str,
    epochs: int = 500,
    fine_tuning_epochs: int = 100,
    timesteps: int = 8,
    batch_size: int = 32,
    noise: str = 'none',
    full_noise: bool = True,
    seed: int = 42,
) -> torch.nn.Module:
    model_type = "clstm"

    best_params = torch.load(f"results/trained_models/{model_type}/params/t1_real.pth")

    model = voronoi_main(
        experiment_name=experiment_name,
        model_type="lstm",
        seed=seed,
        sensor_type="real-random",
        sensor_number=30,
        scaling_type="min-max",
        timesteps=timesteps,
        timesteps_jump=1,
        batch_size=batch_size,
        lr=best_params.get("lr", 1e-4),
        weight_decay=best_params.get("weight_decay", 1e-5),
        epochs=0,
        verbose=False,
        save_model=False,
        early_stopping=False,
        kernel_size=best_params.get("kernel_size", 3),
        use_norm=best_params.get("use_norm", True),
        hidden_channels=best_params.get("hidden_channels", 64),
        n_layers=best_params.get("n_layers", 3),
        noise=noise,
        full_noise=full_noise,
    )

    train_dataset, *_ = load_data(
        sensor_type="real-random",
        sensor_number=30,
        split_mode="monthly",
        scaling_type="min-max",
        combine_train_val=True,
        timesteps=timesteps,
        timesteps_jump=1,
        channel_timesteps=False,
        noise=noise,
        full_noise=full_noise,
        seed=seed
        )
    fine_tuning_dataset, _ = load_real_data(model_type=model_type, sensor_type="real-random", timesteps=timesteps, val_set=False, test_set=False)
    
    dense_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
    sparse_loader = DataLoader(fine_tuning_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)

    seed_everything(seed=seed, verbose=False)

    # Initializing training utils
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=best_params.get("lr", 1e-4), weight_decay=best_params.get("weight_decay", 1e-5))
    scheduler = get_scheduler(optimizer, fine_tuning_epochs)
    loss_fn = get_voronoi_loss_fn()

    for epoch in range(fine_tuning_epochs):
        model.train()

        dense_iter = iter(dense_loader)
        sparse_iter = iter(sparse_loader)

        for batch_dense, batch_sparse in zip_longest(dense_iter, sparse_iter):
            if batch_dense is None or batch_sparse is None:
                continue
            
            obs_sparse, target_sparse, sparse_target_mask = batch_sparse
            obs_sparse, target_sparse, sparse_target_mask = obs_sparse.to(device).float(), target_sparse.to(device).float(), sparse_target_mask.to(device).float()

            # Train the model to reconstruct the sparse observations

            pred_sparse = model(obs_sparse)[:, -1]
            loss_sparse = loss_fn(pred_sparse * sparse_target_mask, target_sparse * sparse_target_mask)

            optimizer.zero_grad()
            loss_sparse.backward()
            optimizer.step()

            # Correct its predictions based on the dense observations

            obs_dense, target_dense = batch_dense
            obs_dense, target_dense = obs_dense.to(device).float(), target_dense.to(device).float()

            torch.clamp(obs_dense, 0, 1, out=obs_dense)

            pred_dense = model(obs_dense)
            loss_dense = loss_fn(pred_dense, target_dense)

            optimizer.zero_grad()
            loss_dense.backward()
            optimizer.step()
        
        safe_scheduler_step(scheduler)

        if (epochs + 1) % 50 == 0:
            logging.info(f"Epoch {epoch + 1}/{epochs}")

    return model


def main(
    experiment_name: str,
    model_types: list[str],
    epochs: int = 500,
    fine_tuning_epochs: int = 100,
    noise: str = 'none',
    full_noise: bool = True,
    seed: int = 42,
):
    for model_type in model_types:

        if model_type == "vunet":
            timesteps = 8
            batch_size = 64

            model = fine_tune_vunet(
                experiment_name=experiment_name,
                epochs=epochs,
                fine_tuning_epochs=fine_tuning_epochs,
                timesteps=timesteps,
                batch_size=batch_size,
                noise=noise,
                full_noise=full_noise,
                seed=seed,
            )

            model.eval()
            with torch.no_grad():
                evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
        elif model_type == "vitae":
            timesteps = 8
            batch_size = 64

            model = fine_tune_vitae(
                experiment_name=experiment_name,
                epochs=epochs,
                fine_tuning_epochs=fine_tuning_epochs,
                timesteps=timesteps,
                batch_size=batch_size,
                noise=noise,
                full_noise=full_noise,
                seed=seed,
            )

            model.eval()
            with torch.no_grad():
                evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
        elif model_type == "clstm":
            timesteps = 8
            batch_size = 32

            model = fine_tune_clstm(
                experiment_name=experiment_name,
                epochs=epochs,
                fine_tuning_epochs=fine_tuning_epochs,
                timesteps=timesteps,
                batch_size=batch_size,
                noise=noise,
                full_noise=full_noise,
                seed=seed,
            )

            model.eval()
            with torch.no_grad():
                evaluate_on_real(model=model, timesteps=timesteps, experiment_name=experiment_name)
        else:
            raise NotImplementedError(f"Model type {model_type} is not implemented.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine tune a model on the real sparse dataset.")
    parser.add_argument("--experiment_name", type=str, default="real_random", help="Name of the experiment")
    parser.add_argument("--model_types", type=str, nargs='+', default=["vcnn", "vunet", "vitae", "clstm"], help="Model types: 'vcnn', 'vunet', 'vitae', or 'clstm'")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
    parser.add_argument("--fine_tuning_epochs", type=int, default=100, help="Number of epochs for fine-tuning")
    parser.add_argument("--noise", type=str, default="none", choices=["none", "gaussian", "time_gaussian", "perlin", "correlated", "white"])
    parser.add_argument("--full_noise", action='store_true', help="Use full noise for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        model_types=args.model_types,
        epochs=args.epochs,
        fine_tuning_epochs=args.fine_tuning_epochs,
        noise=args.noise,
        full_noise=args.full_noise,
        seed=args.seed,
    )