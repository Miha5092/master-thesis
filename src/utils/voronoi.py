import numpy as np
import torch
import os
import torch


def build_voronoi_assign(mask: torch.Tensor) -> torch.Tensor:
    T, C, H, W = mask.shape
    device = mask.device

    res = torch.empty((T, C, H, W), dtype=torch.long, device=device)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    all_pts = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)

    for t in range(T):
        mask_c_hw = mask[t]
        assign_idx = torch.empty((C, H * W), dtype=torch.long, device=device)

        for c in range(C):
            seed_coords = torch.nonzero(mask_c_hw[c] > 0, as_tuple=False)
            if seed_coords.numel() == 0:
                assign_idx[c] = torch.arange(H * W, device=device)
                continue

            diff = all_pts.unsqueeze(1) - seed_coords.unsqueeze(0)
            d2 = (diff ** 2).sum(-1)
            nn = d2.argmin(dim=1)
            seed_flat = seed_coords[:, 0] * W + seed_coords[:, 1]
            assign_idx[c] = seed_flat[nn]

        res[t] = assign_idx.view(C, H, W)

    return res

@torch.no_grad()
def apply_voronoi_gather(Y: torch.Tensor, assign_idx: torch.Tensor) -> torch.Tensor:
    T, C, H, W = Y.shape
    Y_flat = Y.view(T, C, -1)

    idx = assign_idx.view(T, C, -1)

    X_flat = torch.gather(Y_flat, dim=2, index=idx)
    return X_flat.view(T, C, H, W)


def batched_voronoi_tessellation(
    batch_binary_mask: torch.Tensor,
    batch_value_map:   torch.Tensor
) -> torch.Tensor:
    if batch_binary_mask.shape != batch_value_map.shape:
        raise ValueError(f"mask {batch_binary_mask.shape} vs values {batch_value_map.shape}")

    outputs = []
    for mask_c_hw, val_c_hw in zip(batch_binary_mask, batch_value_map):
        out = voronoi_tessellation(mask_c_hw, val_c_hw.unsqueeze(0))  # (1, C, H, W)
        outputs.append(out)

    return torch.cat(outputs, dim=0)  # (B, C, H, W)


def voronoi_tessellation(binary_mask: torch.Tensor, value_map: torch.Tensor) -> torch.Tensor:
    device = value_map.device
    B, C, H, W = value_map.shape

    if binary_mask.shape != (C, H, W):
        raise ValueError(
            f"Binary mask shape {binary_mask.shape} must match value_map channels "
            f"and spatial dimensions {(C, H, W)}"
        )

    result = value_map.clone()

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, H, W) # (B, H, W)

    for c in range(C):
        mask_c = binary_mask[c] # (H, W)

        seed_coords_c = torch.nonzero(mask_c, as_tuple=False) # (N_c, 2)
        num_seeds_c = seed_coords_c.shape[0]

        if num_seeds_c > 0:
            y_seeds_c = seed_coords_c[:, 0].view(num_seeds_c, 1, 1)
            x_seeds_c = seed_coords_c[:, 1].view(num_seeds_c, 1, 1)

            dist_squared_c = (y_grid - y_seeds_c)**2 + (x_grid - x_seeds_c)**2

            closest_seed_idx_c = dist_squared_c.argmin(dim=0)

            closest_seed_coords_c = seed_coords_c[closest_seed_idx_c.view(-1)].view(H, W, 2) # (H, W, 2)

            seed_y_c = closest_seed_coords_c[..., 0]
            seed_x_c = closest_seed_coords_c[..., 1]

            result[:, c, :, :] = value_map[batch_indices, c, seed_y_c, seed_x_c]

    return result


def create_voronoi_map(sensor_number: int):
    d_polair_o3 = np.load('data/d_polair_O3.npy')
    d_polair_pm10 = np.load('data/d_polair_PM10.npy')
    d_polair_pm25 = np.load('data/d_polair_PM25.npy')
    d_polair_no2 = np.load('data/d_polair_NO2.npy')

    all_modalities = np.concatenate((d_polair_o3, d_polair_pm10, d_polair_pm25, d_polair_no2), axis=1)

    assert os.path.exists(f"data/x_coord_{sensor_number}.npy") and os.path.exists(f"data/y_coord_{sensor_number}.npy"), f"There is no know sensor position for {sensor_number} sensors. Please choose between 30, 48, and 108"

    x_coord = np.load(f'data/x_coord_{sensor_number}.npy')
    y_coord = np.load(f'data/y_coord_{sensor_number}.npy')

    mask = np.zeros_like(all_modalities[0, 0])
    mask[x_coord, mask.shape[1] - y_coord] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_modalities = torch.tensor(all_modalities, device=device)

    mask = torch.tensor(mask, device=device)
    voronoi_modalities = voronoi_tessellation(mask, all_modalities)

    voronoi_modalities = voronoi_modalities.cpu()

    np.save(f"data/voronoi_{sensor_number}_O3.npy", voronoi_modalities[:, 0, :, :].unsqueeze(1).numpy())
    np.save(f"data/voronoi_{sensor_number}_PM10.npy", voronoi_modalities[:, 1, :, :].unsqueeze(1).numpy())
    np.save(f"data/voronoi_{sensor_number}_PM25.npy", voronoi_modalities[:, 2, :, :].unsqueeze(1).numpy())
    np.save(f"data/voronoi_{sensor_number}_NO2.npy", voronoi_modalities[:, 3, :, :].unsqueeze(1).numpy())

    np.save(f"data/sensor_mask_{sensor_number}.npy", mask.cpu().numpy())


def create_voronoi_real():
    d_polair_o3 = np.load('data/d_polair_O3.npy')
    d_polair_pm10 = np.load('data/d_polair_PM10.npy')
    d_polair_pm25 = np.load('data/d_polair_PM25.npy')
    d_polair_no2 = np.load('data/d_polair_NO2.npy')

    all_modalities = np.concatenate((d_polair_o3, d_polair_pm10, d_polair_pm25, d_polair_no2), axis=1)

    assert (os.path.exists("data/convert_coord_NO2.npy") and os.path.exists("data/convert_coord_O3.npy") and 
                os.path.exists("data/convert_coord_pm10.npy") and os.path.exists("data/convert_coord_pm25.npy")), "Real sensor position data is missing."
        
    o3_position = np.load("data/convert_coord_O3.npy")
    pm10_position = np.load("data/convert_coord_pm10.npy")
    pm25_position = np.load("data/convert_coord_pm25.npy")
    no2_position = np.load("data/convert_coord_NO2.npy")

    mask = np.zeros_like(all_modalities[0])
    _, h, _ = mask.shape

    mask[0][h - o3_position[:, 1], o3_position[:, 0]] = True
    mask[1][h - pm10_position[:, 1], pm10_position[:, 0]] = True
    mask[2][h - pm25_position[:, 1], pm25_position[:,0]] = True
    mask[3][h - no2_position[:, 1], no2_position[:, 0]] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_modalities = torch.tensor(all_modalities, device=device)

    mask = torch.tensor(mask, device=device)

    o3_voronoi = voronoi_tessellation(mask[0], all_modalities[:, 0, :, 0].unsqueeze(3))
    pm10_voronoi = voronoi_tessellation(mask[1], all_modalities[:, 1, :, 1].unsqueeze(3))
    pm25_voronoi = voronoi_tessellation(mask[2], all_modalities[:, 2, :, 2].unsqueeze(3))
    no2_voronoi = voronoi_tessellation(mask[3], all_modalities[:, 3, :, 3].unsqueeze(3))

    voronoi = torch.cat([o3_voronoi, pm10_voronoi, pm25_voronoi, no2_voronoi], dim=3)

    np.save('data/voronoi_real', voronoi.cpu().numpy())


if __name__ == '__main__':
    create_voronoi_map(30)
    print("Done creating maps for 30 sensors", flush=True)

    create_voronoi_map(48)
    print("Done creating maps for 48 sensors", flush=True)
    
    create_voronoi_map(108)
    print("Done creating maps for 108 sensors", flush=True)

    create_voronoi_real()
    print("Done creating maps for the real sensor placement", flush=True)
    