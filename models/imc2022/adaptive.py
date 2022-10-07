import torch
import torch.nn as nn


class AdaptBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes=2, outplanes=2, stride=1, 
            downsample=None, dilation=1, deformable_groups=1):
        super().__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
                                       [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.transform_matrix_conv = nn.Conv1d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv1d(inplanes, 2, 3, 1, 1, bias=True)
        
        self.conv = nn.Conv1d(20, outplanes, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm1d(outplanes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

 
    def forward(self, x):
        residual = x
        B, _, N = x.shape
        transform_matrix = self.transform_matrix_conv(x)
        transform_matrix = transform_matrix.permute(0, 2, 1).reshape((B*N, 2, 2))
        offset = torch.matmul(transform_matrix, self.regular_matrix)
        offset = offset-self.regular_matrix
        offset = offset.transpose(1, 2).reshape((B, N, 18)).permute(0, 2, 1)
        translation = self.translation_conv(x)

        offset[:,0::2,:] += translation[:,0:1,:]
        offset[:,1::2,:] += translation[:,1:2,:]
 
        x_offset = torch.cat([x, offset], dim=1)
        out = self.conv(x_offset)
        out = self.bn(out)
 
        out = out + residual
        out = self.leaky_relu(out)
 
        return out