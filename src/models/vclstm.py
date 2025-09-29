import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from typing import Callable
from torch.amp import autocast, GradScaler

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super(ConvLSTMCell, self).__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels, 
            out_channels=4*hidden_channels,     # Multiplying by 4 to compute i, f, g, o in one go as channels
            kernel_size=kernel_size, 
            padding=padding,
            bias=True
        )

        self.W_ci = nn.Parameter(torch.Tensor(1, hidden_channels, 1, 1))
        self.W_cf = nn.Parameter(torch.Tensor(1, hidden_channels, 1, 1))
        self.W_co = nn.Parameter(torch.Tensor(1, hidden_channels, 1, 1))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

        nn.init.xavier_uniform_(self.W_ci)
        nn.init.xavier_uniform_(self.W_cf)
        nn.init.xavier_uniform_(self.W_co)
        
    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)

        internal_conv = self.conv(combined)

        xh_i, xh_f, xh_g, xh_o = torch.chunk(internal_conv, 4, dim=1)

        # bias is included from the conv
        i = torch.sigmoid(xh_i + self.W_ci * c_prev) 
        f = torch.sigmoid(xh_f + self.W_cf * c_prev)
        g = torch.tanh(xh_g)
        o = torch.sigmoid(xh_o + self.W_co * c_prev)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        layers = []

        for i in range(num_layers):
            cur_channels = in_channels if i == 0 else hidden_channels

            layers.append(ConvLSTMCell(cur_channels, hidden_channels, kernel_size))

        self.layers = nn.ModuleList(layers)

        padding = kernel_size // 2
        self.output_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, input_tensor):
        device = input_tensor.device
        batch_size, seq_len, _, height, width = input_tensor.shape   # Expected it [B, T, C, H, W]

        h = [torch.zeros(batch_size, self.hidden_channels, height, width, device=input_tensor.device) for _ in self.layers]
        c = [torch.zeros(batch_size, self.hidden_channels, height, width, device=input_tensor.device) for _ in self.layers]

        outputs = torch.empty(batch_size, seq_len, self.out_channels, height, width, device=device)
        for t in range(seq_len):
            x = input_tensor[:, t]  # [T, C, H, W]

            for i, layer in enumerate(self.layers):
                h[i], c[i] = layer(x, h[i], c[i])
                x = h[i]

            outputs[:, t] = self.output_conv(h[-1])

        # outputs = torch.stack(outputs, dim=1)   # [B, T, C, H, W]
        # h = torch.stack(h, dim=0)
        # c = torch.stack(c, dim=0)

        return outputs
    
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

        scaler = GradScaler()

        for batch_idx, (observations, ground_truth) in enumerate(loader):
            observations = observations.to(device, non_blocking=True)
            ground_truth = ground_truth.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                pred = self(observations)
                loss = loss_fn(pred, ground_truth)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # if batch_idx % 50 == 0 and batch_idx > 0:
            #     torch.cuda.empty_cache()
        
            # Delete references to free memory faster
            del observations, ground_truth, pred, loss

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return total_loss / len(loader)
