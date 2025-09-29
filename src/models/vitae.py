import numpy as np
import torch
import torch.nn as nn

from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Callable

from src.models.vcnn import vcnn_tiny
from src.datasets.vitae_dataset import VitaeDataset
from src.utils.evaluation import compute_all_metrics


def get_model(model_type: str, train_dataset: VitaeDataset, patch_size: tuple[int, int], kernel_size: int) -> nn.Module:
    """
    Instantiates the appropriate ViTAE model based on type.

    Args:
        model_type (str): One of ['lite', 'base', 'large', 'unet'].
        train_dataset (VitaeDataset): Training dataset used to infer input shape.
        patch_size (tuple[int, int]): Size of patches for ViTAE.
        kernel_size (int): Convolutional kernel size for UNet variant.

    Returns:
        nn.Module: Initialized PyTorch model of the requested type.
    """

    input_tensor = train_dataset[0][0]
    output_tensor = train_dataset[0][1]

    input_size = input_tensor.shape[1:]

    input_channels = input_tensor.shape[0]
    output_channels = output_tensor.shape[0]

    # Initialize model according to the type.
    if model_type == "lite":
        return vitae_lite(input_size, input_channels, output_channels, patch_size)
    elif model_type == "base":
        return vitae_base(input_size, input_channels, output_channels, patch_size)
    elif model_type == "large":
        return vitae_large(input_size, input_channels, output_channels, patch_size)
    else: # unet
        return vitae_unet(input_size, input_channels, output_channels, patch_size, kernel_size)


def vitae_lite(input_size, in_chans, out_chans, patch_size):
    return ViTAE(
        input_size=input_size, in_chans=in_chans, out_chans=out_chans, patch_size=patch_size,
        enc_chans=16, enc_dim=32, enc_depth=8, enc_num_heads=8, 
        enc_mlp_ratio=4., enc_norm_layer=nn.LayerNorm,
        dec_dims=[16, 16, 16, 16, 16], dec_norm_layer=nn.BatchNorm2d
    )

def vitae_base(input_size, in_chans, out_chans, patch_size):
    return ViTAE(
        input_size=input_size, in_chans=in_chans, out_chans=out_chans, patch_size=patch_size,
        enc_chans=32, enc_dim=64, enc_depth=8, enc_num_heads=8, 
        enc_mlp_ratio=4., enc_norm_layer=nn.LayerNorm,
        dec_dims=[32, 32, 32, 32, 32], dec_norm_layer=nn.BatchNorm2d
    )

def vitae_large(input_size, in_chans, out_chans, patch_size):
    return ViTAE(
        input_size=input_size, in_chans=in_chans, out_chans=out_chans, patch_size=patch_size,
        enc_chans=64, enc_dim=128, enc_depth=8, enc_num_heads=8, 
        enc_mlp_ratio=4., enc_norm_layer=nn.LayerNorm,
        dec_dims=[64, 64, 64, 64, 64], dec_norm_layer=nn.BatchNorm2d
    )

def vitae_unet(input_size, in_chans, out_chans, patch_size, kernel_size):
    return ViTAE(
        input_size=input_size, in_chans=in_chans, out_chans=out_chans, patch_size=patch_size,
        enc_chans=64, enc_dim=128, enc_depth=8, enc_num_heads=8, 
        enc_mlp_ratio=4., enc_norm_layer=nn.LayerNorm,
        dec_dims=[64, 64, 64, 64, 64], dec_norm_layer=nn.BatchNorm2d,
        use_unet='unet', unet_kernel_size=kernel_size
    )


class CNNDecoderBlock(nn.Module):
    def __init__(
        self, in_chans: int, out_chans: int, norm_layer=nn.BatchNorm2d,
    ):
        super(CNNDecoderBlock, self).__init__()

        if norm_layer is not None:
            self.block = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
                norm_layer(out_chans),
                nn.LeakyReLU(0.02, inplace=True),
            )

    def forward(self, x):
        return self.block(x)


class ViTAE(nn.Module):
    def __init__(
        self, input_size: tuple[int, int], in_chans: int, out_chans: int, patch_size: tuple[int, int],
        enc_chans=1, enc_dim=128, enc_depth=8, enc_num_heads=8,
        enc_mlp_ratio=4., enc_norm_layer=nn.LayerNorm,
        dec_dims=[16, 16, 16, 16, 16], dec_norm_layer=nn.BatchNorm2d,
        use_unet: str = None, unet_kernel_size=3,
    ):
        super(ViTAE, self).__init__()

        assert isinstance(input_size, tuple) and len(input_size) == 2, f"input_size {input_size} must be a tuple of 2 integers for (H, W)"
        assert isinstance(patch_size, tuple) and len(patch_size) == 2, f"input_size {patch_size} must be a tuple of 2 integers for (patch_H, patch_W)"
        
        self.input_size = input_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.patch_size = patch_size

        assert input_size[0] % patch_size[0] == 0 and input_size[1] % patch_size[1] == 0, "input_size must be divisible by patch_size"

        # ViT Encoder

        self.enc_chans = enc_chans
        self.patch_embed = PatchEmbed(input_size, patch_size, in_chans, enc_dim) # type: ignore

        self.num_patches = self.patch_embed.num_patches
        self.grid_size = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])

        self.positional_embedding = build_sincos2d_pos_embed(
            feat_shape=[self.grid_size[0], self.grid_size[1]],
            dim=enc_dim
        ).unsqueeze(0)

        self.attention_blocks = nn.ModuleList([
            Block(
                enc_dim, enc_num_heads, enc_mlp_ratio, 
                qkv_bias=True, qk_norm=False,
                proj_drop=0.0, attn_drop=0.0, drop_path=0.0,
                norm_layer=enc_norm_layer
                )
            for _ in range(enc_depth)
        ])
        self.enc_norm = enc_norm_layer(enc_dim)
        self.encoder_conv_out = nn.Conv2d(in_channels=enc_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)

        # CNN or UNet Decoder

        self.decoder_embeddings = nn.Linear(enc_dim, patch_size[0] * patch_size[1] * enc_chans) # Linear layer to expand the dimension of a patch's embedding to the size of a patch

        if use_unet == 'unet':
            self.decoder = vcnn_tiny(enc_chans, dec_dims[-1], unet_kernel_size, use_norm=True)
        else:
            dec_dims = [enc_chans, *dec_dims]
            self.decoder = nn.Sequential(*[
                CNNDecoderBlock(dec_dims[i], dec_dims[i + 1], dec_norm_layer)
                for i in range(len(dec_dims) - 1)
            ])

        self.decoder_conv_out = nn.Conv2d(in_channels=dec_dims[-1], out_channels=out_chans, kernel_size=1, stride=1, padding=0)

        # Initialize weights
        self._init_weights()

    
    def _init_weights(self):

        # Initialize weights of the patch embedding projection layer from the imported module
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the weights of the other PyTorch modules
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        p_h, p_w = self.patch_embed.patch_size
        n_patch_h, n_patch_w = self.grid_size

        x = x.reshape(x.shape[0], n_patch_h, n_patch_w, p_h, p_w, self.enc_chans)                   # Reconstruct the patches from their flattened form
        x = torch.einsum('nhwpqc->nchpwq', x)                                                       # Transpose the dimensions for concatenation
        imgs = x.reshape(x.shape[0], self.enc_chans, n_patch_h * p_h, n_patch_w * p_w)              # Reshape the patches into the original image shape     # type: ignore

        return imgs
    
    def forward_encoder(self, x):
        # Extract patches from the input image
        x = self.patch_embed(x)

        # Add positional encoding to the patches
        x = x + self.positional_embedding.to(x.device)

        # Pass through the transformer
        for block in self.attention_blocks:
            x = block(x)
        x = self.enc_norm(x)

        return x
    
    def forward_decoder(self, x):
        # Expand the dimension of the patch's embedding to the size of a patch
        x = self.decoder_embeddings(x)

        # Reshape into image format for CNN
        imgs = self.unpatchify(x)

        # Pass through the CNN decoder
        x = self.decoder(imgs)

        # Apply the final convolutional layer to reduce the number of channels to the desired output
        pred_enc = self.encoder_conv_out(imgs)
        pred_dec = self.decoder_conv_out(x)

        return pred_enc, pred_dec
    
    def forward(self, x):
        latent = self.forward_encoder(x)
        pred_enc, pred_dec = self.forward_decoder(latent)

        return pred_enc, pred_dec

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

        for observations, ground_truth, mask in loader:
            observations, ground_truth, mask = observations.to(device), ground_truth.to(device), mask.to(device)

            pred_enc, pred_dec = self(observations)
            loss = loss_fn(pred_enc, pred_dec, mask, ground_truth)

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
        advanced_statistics (bool): If True, computes additional statistics.

    Returns:
        Tuple containing:
            - float: Average loss
            - list: Relative errors for each batch
            - list: SSIM values
            - list: PSNR values
            - list: Local errors
            - list: Input observations
            - list: Ground truth labels
            - list: Encoder predictions
            - list: Decoder predictions
    """

    model.eval()
    avg_loss = 0
    all_obs, all_gt, enc_preds, dec_preds = [], [], [], []
    relative_errors, ssims, psnrs, local_errors = [], [], [], []

    with torch.no_grad():
        for observations, ground_truth, mask in loader:
            observations, ground_truth, mask = observations.to(device), ground_truth.to(device), mask.to(device)
            pred_enc, pred_dec = model(observations)

            if loss_fn:
                loss = loss_fn(pred_enc, pred_dec, mask, ground_truth)
                avg_loss += loss.item()

            if advanced_statistics:
                batch_relative_error, batch_ssim, batch_psnr, batch_local_errors = compute_all_metrics(ground_truth, pred_dec)
                relative_errors.append(batch_relative_error)
                ssims.append(batch_ssim)
                psnrs.append(batch_psnr)
                local_errors.append(batch_local_errors)

                all_obs.append(observations.cpu().numpy())
                all_gt.append(ground_truth.cpu().numpy())
                enc_preds.append(pred_enc.cpu().numpy())
                dec_preds.append(pred_dec.cpu().numpy())

    if advanced_statistics:
        relative_errors = np.concatenate(relative_errors, axis=0)
        ssims = np.concatenate(ssims, axis=0)
        psnrs = np.concatenate(psnrs, axis=0)
        local_errors = np.concatenate(local_errors, axis=0)
        
        all_obs = np.concatenate(all_obs, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        enc_preds = np.concatenate(enc_preds, axis=0)
        dec_preds = np.concatenate(dec_preds, axis=0)

    avg_loss /= len(loader)
    return avg_loss, relative_errors, ssims, psnrs, local_errors, all_obs, all_gt, enc_preds, dec_preds
    