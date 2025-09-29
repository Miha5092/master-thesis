import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from typing import Callable

from src.models.vclstm import ConvLSTM
from src.utils.evaluation import compute_all_metrics


def get_model(model_type: str, train_dataset: Dataset, kernel_size: int, use_norm: bool, hidden_channels: int, n_layers: int) -> nn.Module:
    """
    Instantiates the appropriate VCNN model based on type.

    Args:
        model_type (str): One of ['tiny', 'lite', 'base', 'large', 'classic'].
        train_dataset (VoronoiDataset): Training dataset used to infer input shape.
        kernel_size (int): Size of the convolutional kernel.
        use_norm (bool): Whether to use normalization layers in the model.
        hidden_channels (int): Number of hidden channels for ConvLSTM.
        n_layers (int): Number of layers for ConvLSTM.

    Returns:
        nn.Module: Initialized PyTorch model of the requested type.
    """

    input_tensor = train_dataset[0][0]
    output_tensor = train_dataset[0][1]

    input_channels = input_tensor.shape[-3]
    output_channels = output_tensor.shape[-3]

    # Initialize model based on type.
    if model_type == "tiny":
        return vcnn_tiny(input_channels, output_channels, kernel_size, use_norm)
    elif model_type == "lite":
        return vcnn_lite(input_channels, output_channels, kernel_size, use_norm)
    elif model_type == "base":
        return vcnn_base(input_channels, output_channels, kernel_size, use_norm)
    elif model_type == "large":
        return vcnn_large(input_channels, output_channels, kernel_size, use_norm)
    elif model_type == "classic":
        return vcnn_classic(input_channels, output_channels)
    else:   # ConvLSTM
        model = ConvLSTM(input_channels, hidden_channels, output_channels, kernel_size, n_layers)
        model = torch.compile(model, mode="max-autotune")
        return model


def vcnn_classic(in_chans: int, out_chans: int):
    return VCNN_classic(in_chans, out_chans)

def vcnn_tiny(in_chans: int, out_chans: int, kernel_size: int = 3, use_norm: bool = True):
    return VCNN(in_chans, out_chans, 1, kernel_size, 32, use_norm = use_norm)   

def vcnn_lite(in_chans: int, out_chans: int, kernel_size: int = 3, use_norm: bool = True):
    return VCNN(in_chans, out_chans, 1, kernel_size, use_norm = use_norm)

def vcnn_base(in_chans: int, out_chans: int, kernel_size: int = 3, use_norm: bool = True):
    return VCNN(in_chans, out_chans, 3, kernel_size, use_norm = use_norm)

def vcnn_large(in_chans: int, out_chans: int, kernel_size: int = 3, use_norm: bool = True):
    return VCNN(in_chans, out_chans, 5, kernel_size, use_norm = use_norm)


def create_convs(in_channels, out_channels, block_depth, kernel_size, padding, use_norm):
    if use_norm:
        convs = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02, inplace=True),
        ]
    else:
        convs = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.02, inplace=True),
        ]

    for _ in range(block_depth - 1):
        convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if use_norm:
            convs.append(out_channels)
        convs.append(nn.LeakyReLU(0.02, inplace=True))

    return convs


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_depth: int, kernel_size: int  = 3, use_norm: bool = True):
        super(DownBlock, self).__init__()

        padding = kernel_size // 2

        self.convs = nn.Sequential(*create_convs(in_channels, out_channels, block_depth, kernel_size, padding, use_norm))

        self.downscale = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.convs(x)

        downscaled_x = self.downscale(x)

        return downscaled_x, x


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_depth: int, kernel_size: int = 3, use_norm: bool = True):
        super(Bottleneck, self).__init__()
        
        padding = kernel_size // 2

        self.convs = nn.Sequential(*create_convs(in_channels, out_channels, block_depth, kernel_size, padding, use_norm))

    def forward(self, x: torch.Tensor):
        return self.convs(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, incoming_channels: int, out_channels: int, block_depth: int, kernel_size: int = 3, output_padding: tuple = (0, 0), use_norm: bool = True):
        super(UpBlock, self).__init__()

        padding = kernel_size // 2

        self.upscale = nn.ConvTranspose2d(
            in_channels=incoming_channels,
            out_channels=incoming_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=output_padding
        )

        self.convs = nn.Sequential(*create_convs(in_channels, out_channels, block_depth, kernel_size, padding, use_norm))


    def forward(self, x: torch.Tensor, x_skip: torch.Tensor):
        x_upscaled = self.upscale(x)

        x = torch.cat([x_upscaled, x_skip], dim=1)

        x = self.convs(x)

        return x


class VCNN(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            block_depth: int = 1, 
            kernel_size: int = 3, 
            hidden_channels: int = 64,
            use_norm: bool = True,
            ):
        super(VCNN, self).__init__()

        assert kernel_size % 2 == 1 , f"Kernel size {kernel_size} must be an odd number"

        padding = kernel_size // 2

        self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding)

        self.down1 = DownBlock(hidden_channels, hidden_channels * 2, block_depth, kernel_size, use_norm)    # 75x110 -> 37x55
        self.down2 = DownBlock(hidden_channels * 2, hidden_channels * 4, block_depth, kernel_size, use_norm)   # 37x55 -> 18x27
        self.down3 = DownBlock(hidden_channels * 4, hidden_channels * 4, block_depth, kernel_size, use_norm)   # 18x27 -> 9x13

        self.bottleneck = Bottleneck(hidden_channels * 4, hidden_channels * 4, block_depth, kernel_size, use_norm)     # 9x13 -> 9x13

        self.up1 = UpBlock(hidden_channels * 4 + hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, block_depth, kernel_size, output_padding=(0, 1), use_norm=use_norm) # 9x13 -> 18x27
        self.up2 = UpBlock(hidden_channels * 4 + hidden_channels * 4, hidden_channels * 4, hidden_channels * 2, block_depth, kernel_size, output_padding=(1, 1), use_norm=use_norm)  # 18x27 -> 37x55
        self.up3 = UpBlock(hidden_channels * 2 + hidden_channels * 2, hidden_channels * 2, hidden_channels, block_depth, kernel_size, output_padding=(1, 0), use_norm=use_norm)  # 37x55 -> 75x110

        self.final_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor):
        x = self.initial_conv(x)

        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)

        x = self.bottleneck(x)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.final_conv(x)

        return x
    
    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable,
        device: torch.device,
    ) -> float:
        """
        Trains the model for one epoch over the given DataLoader.

        Args:
            model (nn.Module): Model to be trained.
            loader (DataLoader): Training data loader.
            optimizer (Optimizer): Optimizer for weight updates.
            loss_fn (Callable): Loss function.
            device (torch.device): Device to perform computation on.

        Returns:
            float: Average training loss over the epoch.
        """
            
        self.train()
        total_loss = 0.0

        for observations, ground_truth in loader:
            observations, ground_truth = observations.to(device), ground_truth.to(device)

            torch.clamp(observations, 0, 1, out=observations)

            pred = self(observations)
            loss = loss_fn(pred, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


class VCNN_classic(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            ):
        super(VCNN_classic, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(48, out_channels, 7, 1, 3), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.convs(x)
    
    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable,
        device: torch.device,
    ) -> float:
        """
        Trains the model for one epoch over the given DataLoader.

        Args:
            model (nn.Module): Model to be trained.
            loader (DataLoader): Training data loader.
            optimizer (Optimizer): Optimizer for weight updates.
            loss_fn (Callable): Loss function.
            device (torch.device): Device to perform computation on.

        Returns:
            float: Average training loss over the epoch.
        """
            
        self.train()
        total_loss = 0.0

        for observations, ground_truth in loader:
            observations, ground_truth = observations.to(device), ground_truth.to(device)

            torch.clamp(observations, 0, 1, out=observations)

            pred = self(observations)
            loss = loss_fn(pred, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    advanced_statistics: bool,
) -> tuple[float, list, list, list, list, list, list, list, list]:
    """
    Evaluates the model on the given dataset without updating weights.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Evaluation DataLoader.
        loss_fn (Callable): Loss function to compute evaluation loss.
        device (torch.device): Device to perform inference on.
        advanced_statistics (bool): If True, computes additional metrics.

    Returns:
        Tuple containing:
            - float: Average loss
            - list: Relative errors for each batch
            - list: SSIM values
            - list: PSNR values
            - list: Local errors
            - list: Input observations
            - list: Ground truth labels
            - list: Predictions
    """

    model.eval()
    avg_loss = 0
    all_obs, all_gt, preds = [], [], []
    relative_errors, ssims, psnrs, local_errors = [], [], [], []

    with torch.no_grad():
        for observations, ground_truth in loader:
            observations, ground_truth = observations.to(device), ground_truth.to(device)
            pred = model(observations)

            if loss_fn:
                loss = loss_fn(pred, ground_truth)
                avg_loss += loss.item()

            if advanced_statistics:
                batch_relative_error, batch_ssim, batch_psnr, batch_local_errors = compute_all_metrics(ground_truth, pred)
                relative_errors.append(batch_relative_error)
                ssims.append(batch_ssim)
                psnrs.append(batch_psnr)
                local_errors.append(batch_local_errors)

                all_obs.append(observations.cpu().numpy())
                all_gt.append(ground_truth.cpu().numpy())
                preds.append(pred.cpu().numpy())

    if advanced_statistics:
        relative_errors = np.concatenate(relative_errors, axis=0)
        ssims = np.concatenate(ssims, axis=0)
        psnrs = np.concatenate(psnrs, axis=0)
        local_errors = np.concatenate(local_errors, axis=0)
        
        all_obs = np.concatenate(all_obs, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        preds = np.concatenate(preds, axis=0)

    avg_loss /= len(loader)
    return avg_loss, relative_errors, ssims, psnrs, local_errors, all_obs, all_gt, preds
    