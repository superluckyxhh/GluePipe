import torch
import torch.nn as nn
import kornia.feature as KF


def get_lafs_scale(lafs):
    """
    Args:
        lafs (_type_): [B, N, 2, 3]
    Returns:
        scale: [B, N, 1, 1]
    """
    eps = 1e-10
    out = lafs[..., 0:1, 0:1] * lafs[..., 1:2, 1:2] - lafs[..., 1:2, 0:1] * lafs[..., 0:1, 1:2] + eps
    return out.abs().sqrt()


class Geometric_Oritation_Embedding(nn.Module):
    def __init__(self, method='orientation'):
        super().__init__()
        self.method = method
        
    def forward(self, lafs):
        if self.method == 'orientation':
            scale = get_lafs_scale(lafs).squeeze(-1)
            return torch.flip(lafs[..., 0, :-1], dims=(-1,)) / scale
    
    @property
    def side_info_dim(self):
        return 2