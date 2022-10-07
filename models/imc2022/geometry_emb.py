import torch
import torch.nn as nn
import numpy as np


def pairwise_distance(points0, points1, normlized):
    """Compute a pairwise distance of points

    Args:
        points0 (_type_): (B, N, 2)
        points1 (_type_): (B, N, 2)
    """
    # [B, N, 2] @ [B, 2, N] --> [B, N, N]
    xy = torch.matmul(points0, points1.transpose(-1, -2)) 
    if normlized:
        sq_dist = 2.0 - 2.0 * xy
    else:
        # [B, N, 2] --> [B, N] --> [B, N, 1]
        x2 = torch.sum(x ** 2, dim=-1).unsqueeze(-1)
        # [B, N, 2] --> [B, N] --> [B, 1, N]
        y2 = torch.sum(y ** 2, dim=-1).unsqueeze(-2)    
        sq_dist = x2 - 2 * xy + y2
    sq_dist = sq_dist.clamp(min=0.0)
    
    return sq_dist


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()

        return embeddings
    
    
class GeometricEmbedding(nn.Module):
    def __init__(self, sigma_d=1, points_normlize=True):
        super().__init__()
        self.sigma_d = sigma_d
        self.points_normlize = points_normlize
        self.sin_embedding = SinusoidalPositionalEmbedding(2)
        
    @torch.no_grad()
    def get_embedding_indices(self, points0, points1):
        batch_size, num_point, _ = points0.shape

        dist_map = torch.sqrt(pairwise_distance(points0, points1, self.points_normlize))  # (B, N, N)
        d_indices = dist_map / self.sigma_d

        return d_indices
    
    def forward(self, points0, points1):
        d_indices = self.get_embedding_indices(points0, points1)
        d_embeddings = self.sin_embedding(d_indices)
        
        points0_emb = torch.einsum("bnc,bnmc->bnc", points0, d_embeddings)
        points1_emb = torch.einsum("bmc,bmnc->bmc", points1, d_embeddings.permute(0, 2, 1, 3))
        
        return points0_emb, points1_emb