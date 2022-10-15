
import torch
import torch.nn as nn
import torch.nn.functional as F


class Geometric_Position_Adaptive(nn.Module):
    def __init__(
        self, d_model,
        eps=1e-6,
    ):
        super().__init__()
        regular_pts = torch.tensor(
            [
                [-1, -1, -1, 0, 0, 0, 1, 1, 1],\
                [-1, 0, 1, -1, 0, 1, -1, 0, 1]
            ]
        )
        self.register_buffer('regular_pts', regular_pts.float())
        self.eps = eps
        self.pi = torch.pi
        self.rotation = 0.
        self.transform_rotation = nn.Conv1d(d_model, 4, 3, 1, 1, bias=True)
        self.transform_translation = nn.Conv1d(d_model, 2, 3, 1, 1, bias=True)

        # Position        
        self.pos_conv = nn.Conv1d(20, 2, kernel_size=3, padding=1, dilation=1)
        self.pos_bn = nn.BatchNorm1d(2)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)


    def compute_offset(self, feats):
        """
            feats: shape [B, 256, N]
        """
        B, _, N = feats.shape
        transform_matrix = self.transform_rotation(feats)
        transform_matrix = transform_matrix.permute(0, 2, 1).reshape((B*N,2,2))
        tranform_regular_pts = torch.matmul(transform_matrix, self.regular_pts)

        # The offset in the x&y direction
        offset = tranform_regular_pts - self.regular_pts
        offset = offset.transpose(1, 2).reshape((B, N, 18)).permute(0, 2, 1)
    
        translation = self.transform_translation(feats)
  
        offset[:, 0::2, :] += translation[:, 0:1, :]
        offset[:, 1::2, :] += translation[:, 1:2, :]
       
        return offset
        
        
    def forward(self, pts, feats):
        b, _, n = pts.size()
        offset = self.compute_offset(feats)
        
        # Adjust the position of the points according to the offset
        pts = torch.cat([pts, offset], dim=1)
        pts = self.pos_conv(pts)
        pts = self.pos_bn(pts)
        pts = self.leaky_relu(pts)
            
        return pts