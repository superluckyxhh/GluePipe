
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricEmbedding(nn.Module):
    def __init__(
        self, d_model, 
        geo_position=True, 
        geo_orientation=True,
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
        self.transform_rotation = nn.Conv1d(d_model, 4, 3, 1, 1, bias=True)
        self.transform_translation = nn.Conv1d(d_model, 2, 3, 1, 1, bias=True)

        self.geo_position = geo_position
        self.geo_orientation = geo_orientation
     
        # Position        
        if self.geo_position:
            self.pos_conv = nn.Conv1d(20, 2, kernel_size=3, padding=1, dilation=1)
            self.pos_bn = nn.BatchNorm1d(2)
            # self.offset_conv = nn.Conv1d(18, 2, kernel_size=3, padding=1, dilation=1)
            # self.offset_bn = nn.BatchNorm1d(2)
        
        # Orientation
        if self.geo_orientation:
            self.conv = nn.Conv1d(2, 2, kernel_size=3, padding=1, dilation=1)
            self.bn = nn.BatchNorm1d(2)
        
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


    def anglerad_to_rotation_matrix(self, rad):
        b, _, n = rad.size()
        cos_a = torch.cos(rad).transpose(1, 2).contiguous()
        sin_a = torch.sin(rad).transpose(1, 2).contiguous()
        matrix = torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1)
        matrix = matrix.view(b, n, 2, 2).contiguous()
        return matrix
        
        
    def forward(self, pts, feats):
        b, _, n = pts.size()
        offset = self.compute_offset(feats)
        if self.geo_position:
            # Adjust the position of the points according to the offset
            pts = torch.cat([pts, offset], dim=1)
            
            # main_offset = self.offset_conv(offset)
            # main_offset = self.offset_bn(main_offset)
            # pts = pts + main_offset
            pts = self.pos_conv(pts)
            pts = self.pos_bn(pts)
            pts = self.leaky_relu(pts)
            
        if self.geo_orientation:
            tan_theta = (offset[:, 1::2, :] + 1e-6) / (offset[:, 0::2, :] + self.eps)
            rad = torch.atan(tan_theta)
            main_rad, _ = torch.max(rad, dim=1, keepdim=True)
            rotation = self.anglerad_to_rotation_matrix(main_rad)
            
            # [b, 2, n] --> [b, n, 2]
            pts = pts.transpose(1, 2).contiguous()
            # [b, n, 2] @ [b, n, 2, 2] --> [b, n, 2]
            pts = torch.einsum('bnx,bnxy->bny', pts, rotation)
            # [b, n, 2] --> [b, 2, n]
            pts = pts.transpose(1, 2).contiguous()
            
            pts = self.conv(pts)
            pts = self.bn(pts)

        return pts