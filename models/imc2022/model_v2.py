from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
import torch
from torch import nn

from models.imc2022.geometry_adaptive_pos import Geometric_Position_Adaptive
from models.imc2022.geometric_oritation_emb import Geometric_Oritation_Embedding
from models.imc2022.feedforward import MLPPositionalEncoding
    
    
def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


# class KeypointEncoder(nn.Module):
#     """ Joint encoding of visual appearance and location using MLPs"""
#     def __init__(self, feature_dim: int, layers: List[int]) -> None:
#         super().__init__()
#         self.encoder = MLP([3] + layers + [feature_dim])
#         nn.init.constant_(self.encoder[-1].bias, 0.0)

#     def forward(self, kpts, scores):
#         inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
#         return self.encoder(torch.cat(inputs, dim=1))
    

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class IMCNet(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
                       
        self.geo_rot_emb = Geometric_Oritation_Embedding()
        self.geo_adapt_pos = Geometric_Position_Adaptive(
            self.config['descriptor_dim']
        )
        self.config['positional_encoding']['side_info_size'] = \
            self.geo_rot_emb.side_info_dim + 1  # plus 1 for responses
            
        self.position_encoding = MLPPositionalEncoding(**self.config['positional_encoding'])
        # self.kenc = KeypointEncoder(
        #     self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        
        self.log_response = self.config.get('log_scores', False)
        self.residual = self.config.get('residual', False)
        if self.residual:
            self.mix_coefs = nn.parameter.Parameter(torch.zeros(self.config['descriptor_dim'], 1))
            
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)


    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        desc0, desc1 = desc0.transpose(1, 2).contiguous(), desc1.transpose(1, 2).contiguous()
        image0_size, image1_size = data['image0_size'][::-1], data['image1_size'][::-1]
        
        scores0, scores1 = data['scores0'].unsqueeze(-1), data['scores1'].unsqueeze(-1) 
        lafs0, lafs1 = data['lafs0'], data['lafs1']
        kpts0, kpts1 = lafs0[:, :, :, -1], lafs1[:, :, :, -1]
        
        # Pre-Process : lafs --> side_info
        side_info0 = self.geo_rot_emb(lafs0)
        side_info1 = self.geo_rot_emb(lafs1)
        
        if self.log_response:
            scores0 = (scores0 + 0.1).log()
            scores1 = (scores1 + 0.1).log()
            
        side_info0 = torch.cat([scores0, side_info0], dim=-1)
        side_info1 = torch.cat([scores1, side_info1], dim=-1)
        
        # Step1. Keypoint normalization      
        kpts0, kpts1 = normalize_keypoints(kpts0, image0_size), normalize_keypoints(kpts1, image1_size)
        kpts0, kpts1 = kpts0.transpose(1, 2).contiguous(), kpts1.transpose(1, 2).contiguous()
        kpts0 = self.geo_adapt_pos(kpts0, desc0)
        kpts1 = self.geo_adapt_pos(kpts1, desc1)
        kpts0, kpts1 = kpts0.transpose(1, 2).contiguous(), kpts1.transpose(1, 2).contiguous()
        
        # Keypoint MLP encoder (side_info & keypoints)
        pe0, pe1 = self.position_encoding(kpts0, side_info0), self.position_encoding(kpts1, side_info1)
        
        desc0, desc1 = desc0 + pe0, desc1 + pe1         

        # Multi-layer Transformer network.
        gdesc0, gdesc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        gdesc0, gdesc1 = self.final_proj(gdesc0), self.final_proj(gdesc1)

        if self.residual:
            alpha = torch.sigmoid(self.mix_coefs)
            gdesc0 = alpha * gdesc0 + (1.0 - alpha) * desc0
            gdesc1 = alpha * gdesc1 + (1.0 - alpha) * desc1
            
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', gdesc0, gdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        return {
            'context_descriptors0': gdesc0,
            'context_descriptors1': gdesc1,
            'scores': scores
        }
    
    
    def inference(self, data):
        # run feature extractor on both images
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        desc0, desc1 = desc0.transpose(1, 2).contiguous(), desc1.transpose(1, 2).contiguous()
        image0_size, image1_size = data['image0_size'][::-1], data['image1_size'][::-1]
        
        scores0, scores1 = data['scores0'].unsqueeze(-1), data['scores1'].unsqueeze(-1) 
        lafs0, lafs1 = data['lafs0'], data['lafs1']
        unnorm_kpts0, unnorm_kpts1 = lafs0[:, :, :, -1], lafs1[:, :, :, -1]
        
        # Pre-Process : lafs --> side_info
        side_info0 = self.geo_rot_emb(lafs0)
        side_info1 = self.geo_rot_emb(lafs1)
        
        if self.log_response:
            scores0 = (scores0 + 0.1).log()
            scores1 = (scores1 + 0.1).log()
            
        side_info0 = torch.cat([scores0, side_info0], dim=-1)
        side_info1 = torch.cat([scores1, side_info1], dim=-1)

        # Step1. Keypoint normalization      
        kpts0, kpts1 = normalize_keypoints(unnorm_kpts0, image0_size), normalize_keypoints(unnorm_kpts1, image1_size)
        kpts0, kpts1 = kpts0.transpose(1, 2).contiguous(), kpts1.transpose(1, 2).contiguous()
        kpts0 = self.geo_adapt_pos(kpts0, desc0)
        kpts1 = self.geo_adapt_pos(kpts1, desc1)
        kpts0, kpts1 = kpts0.transpose(1, 2).contiguous(), kpts1.transpose(1, 2).contiguous()
        
        # Keypoint MLP encoder (side_info & keypoints)
        pe0, pe1 = self.position_encoding(kpts0, side_info0), self.position_encoding(kpts1, side_info1)
        desc0, desc1 = desc0 + pe0, desc1 + pe1 

        # Multi-layer Transformer network.
        gdesc0, gdesc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        gdesc0, gdesc1 = self.final_proj(gdesc0), self.final_proj(gdesc1)

        if self.residual:
            alpha = torch.sigmoid(self.mix_coefs)
            gdesc0 = alpha * gdesc0 + (1.0 - alpha) * desc0
            gdesc1 = alpha * gdesc1 + (1.0 - alpha) * desc1
            
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', gdesc0, gdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))

        return {
            'keypoints0': unnorm_kpts0,
            'keypoints1': unnorm_kpts1,
            'matches0': indices0,  # use -1 for invalid match
            'matching_scores0': mscores0,
        }