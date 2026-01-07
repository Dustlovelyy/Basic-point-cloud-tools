import numpy as np
import torch

def farthest_point_sample(points, lengths=None, K=512, random_start_point=False):
    """
    Batch farthest point sampling (FPS)

    Args:
        points: torch.Tensor, input point cloud [B, N, 3]
        lengths: (optional) torch.Tensor, actual length of each point cloud [B]
        K: int, number of points to sample
        random_start_point: bool, whether to randomly select the starting point

    Returns:
        selected_points: torch.Tensor, sampled point cloud [B, K, 3]
        selected_indices: torch.Tensor, indices of sampled points [B, K]
    """
    device = points.device
    B, N, C = points.shape

    # Ensure K does not exceed the number of points
    K = min(K, N)

    # Initialize result tensors
    selected_indices = torch.zeros(B, K, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10

    # Select the starting point
    if random_start_point:
        start_indices = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    else:
        start_indices = torch.zeros(B, dtype=torch.long, device=device)

    selected_indices[:, 0] = start_indices
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    # Iteratively select the farthest point
    for k in range(1, K):
        # Get the last selected point
        last_selected = points[batch_indices, selected_indices[:, k-1]]

        # Calculate distances from all points to the last selected point
        dist_to_last = torch.sum((points - last_selected.unsqueeze(1)) ** 2, dim=-1)

        # Update minimum distance
        distance = torch.min(distance, dist_to_last)

        # Handle valid lengths
        if lengths is not None:
            mask = torch.arange(N, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            distance.masked_fill_(mask, -1e10)

        # Select the point with the maximum distance as the next point
        selected_indices[:, k] = torch.argmax(distance, dim=1)

    # Extract selected points
    selected = torch.gather(points, 1, selected_indices.unsqueeze(-1).expand(-1, -1, 3))

    return selected, selected_indices

# Voxel-based FPS
def farthest_voxel_points_samplings(voxel_point_dir, vox_density_dir):
    all_sampled_points = []

    for voxel, points in voxel_point_dir.items():
        density = vox_density_dir.get(voxel, 1)

        points_np = np.array(points)
        points_count = points_np.shape[0]

        # Change sampling rate
        default_voxel_downsample_rate = 0.5
        default_samples_per_voxel = default_voxel_downsample_rate * points_count

        # Segmented sampling
        if 0 <= density <= 0.3:
            counted_density = 1
        elif 0.3 < density <= 0.7:
            counted_density = 0.4
        else:
            counted_density = 0.3

        num_samples_per_voxel = int(default_samples_per_voxel * counted_density)

        sampled_points = farthest_point_sample(points_np, num_samples_per_voxel)
        all_sampled_points.append(sampled_points)

    # Merge sampled points from all voxels
    all_sampled_points = np.vstack(all_sampled_points)
    # print("sampled and returned with the shape of : ", {all_sampled_points.shape})
    return all_sampled_points

def farthest_point_sample_batch(xyz, npoint):
    """
    Batch GPU version of farthest point sampling.
    Args:
        xyz: (B, N, 3) torch.Tensor, float32, on GPU
        npoint: int, number of samples
    Returns:
        centroids: (B, npoint) torch.LongTensor, indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids