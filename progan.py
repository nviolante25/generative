import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def pixel_norm(x, eps=10e-8):
    return x / (torch.mean(x ** 2, dim=1, keepdim=True) + eps).sqrt()

class PixelNorm2d(nn.Module):
    def __init__(self, eps=10e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return pixel_norm(x, self.eps)
    

class EqualizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        nn.init.normal_(self.weight, 0.0, 1.0)
        self.weight_scale = math.sqrt(2.0 / in_features)
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        return F.linear(input, self.weight * self.weight_scale, self.bias)


class EqualizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros', 
        device=None, 
        dtype=None
    
    ):
        super().__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode, 
            device=device, 
            dtype=dtype,
        )
        nn.init.normal_(self.weight, 0.0, 1.0)
        self.weight_scale = math.sqrt(2.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.weight_scale, self.bias)


class GeneratorInitBlock(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        h = w = 4
        self.fc0 = EqualizedLinear(z_dim, z_dim * h * w)
        self.reshape = Rearrange("b (c h w) -> b c h w", c=z_dim, h=h, w=w)
        self.conv0 = EqualizedConv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.to_rgb = EqualizedConv2d(in_channels=z_dim, out_channels=3, kernel_size=1)


    def forward(self, x):
        x = pixel_norm(F.leaky_relu(self.reshape(self.fc0(x)), 0.2))
        x = pixel_norm(F.leaky_relu(self.conv0(x), 0.2))
        return x

class GeneratorUpBlock(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        assert len(channels_list) == 3
        in_channels, hidden_channels, out_channels = channels_list
        self.conv0 = EqualizedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv1 = EqualizedConv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.to_rgb = EqualizedConv2d(out_channels, out_channels=3, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = pixel_norm(F.leaky_relu(self.conv0(x), 0.2))
        x = pixel_norm(F.leaky_relu(self.conv1(x), 0.2))
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_resolution):
        super().__init__()
        assert img_resolution % 2 == 0
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        

        channels_per_block = [
            [z_dim, z_dim, z_dim],                      # 8x8
            [z_dim, z_dim, z_dim],                      # 16x16
            [z_dim, z_dim, z_dim],                      # 32x32
            [z_dim, z_dim // 2, z_dim // 2],            # 64x64
            [z_dim // 2, z_dim // 4, z_dim // 4],       # 128x128
            [z_dim // 4, z_dim // 8, z_dim // 8],       # 256x256
            [z_dim // 8, z_dim // 16, z_dim // 16],     # 512x512   
            [z_dim // 16, z_dim // 32, z_dim // 32],    # 1024X1024
        ]

        # Create named attributes: b4x4, b8x8,...,b1024x1024
        self.add_module("b4x4", GeneratorInitBlock(z_dim))
        num_up_blocks = int(math.log2(img_resolution) - 2)
        for i in range(1, num_up_blocks + 1):
            res = self.idx2resolution(i)
            self.add_module(f"b{res}x{res}", GeneratorUpBlock(channels_per_block[i]))

        self.num_blocks = num_up_blocks + 1

    def forward(self, z, alpha=1.0, last_block=None):
        """
        Input:
            z: [batch_size, z_dim]
            alpha: default no faded transition

        Output:
            x: [batch_size, 3, img_resolution, img_resolution]
        """
        assert z.shape[1] == self.z_dim
        if last_block is None:
            last_block = self.num_blocks - 1
        assert last_block <= self.num_blocks - 1

        x = z
        for i in range(last_block + 1):
            res = self.idx2resolution(i)
            block = self.get_submodule(f"b{res}x{res}")
            x_prev = x
            x = block(x)
        to_rgb = self.get_submodule(f"b{res}x{res}.to_rgb")
        rgb = to_rgb(x)

        # Faded skip connection during training
        if last_block > 0 and alpha < 1.0:
            x_prev = F.interpolate(x_prev, scale_factor=2.0, mode="nearest")
            to_rgb_prev = self.get_submodule(f"b{res//2}x{res//2}.to_rgb")
            rgb = alpha * rgb + (1.0 - alpha) * to_rgb_prev(x_prev)

        return rgb

    @staticmethod
    def idx2resolution(idx):
        return int(2 ** (idx + 2))

    def sample_z(self, batch_size):
        return torch.randn((batch_size, self.z_dim), dtype=torch.float32)
    


gen = Generator(512, 256)
z = gen.sample_z(2)
x = gen(z)
print()