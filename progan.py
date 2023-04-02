import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


class PixelNorm2d(nn.Module):
    def __init__(self, eps=10e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + self.eps).sqrt()
    

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


class GeneratorBlock(nn.Module):
    def __init__(self, channels_list):
        super().__init__()
        assert len(channels_list) == 3
        in_channels, hidden_channels, out_channels = channels_list
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            EqualizedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            EqualizedConv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_resolution):
        super().__init__()
        assert img_resolution % 2 == 0
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        
        init_block = nn.Sequential(
            EqualizedLinear(z_dim, z_dim * 16),
            Rearrange("b (c h w) -> b c h w", c=z_dim, h=4, w=4),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
            EqualizedConv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            PixelNorm2d(),
        )
        init_to_rgb = EqualizedConv2d(in_channels=z_dim, out_channels=3, kernel_size=1)

        channels_per_block = [
            [512, 512, 512], # 8x8
            [512, 512, 512], # 16x16
            [512, 512, 512], # 32x32
            [512, 256, 256], # 64x64
            [256, 128, 128], # 128x128
            [128, 64, 64],   # 256x256
            [64, 32, 32],    # 512x512   
            [32, 16, 16],    # 1024X1024
        ]
        self.gen_blocks = nn.ModuleList()
        self.gen_blocks.append(nn.ModuleDict({"up": init_block, "to_rgb": init_to_rgb}))

        num_up_blocks = int(math.log2(img_resolution) - 2)
        for i in range(num_up_blocks):
            up = GeneratorBlock(channels_per_block[i])
            to_rgb = EqualizedConv2d(in_channels=channels_per_block[i][-1], out_channels=3, kernel_size=1)
            self.gen_blocks.append(nn.ModuleDict({"up": up, "to_rgb": to_rgb}))

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
            last_block = len(self.gen_blocks) - 1
        assert last_block <= len(self.gen_blocks) - 1

        x = z
        for block in self.gen_blocks[:last_block+1]:
            x_prev = x
            x = block["up"](x)

        # Faded skip connection
        to_rgb = self.gen_blocks[last_block]["to_rgb"]
        rgb = to_rgb(x)
        if last_block > 0:
            x_prev = F.interpolate(x_prev, scale_factor=2.0, mode="nearest")
            to_rgb_prev = self.gen_blocks[last_block - 1]["to_rgb"]
            rgb = alpha * rgb + (1.0 - alpha) * to_rgb_prev(x_prev)


        if last_block == len(self.gen_blocks) - 1:
            assert (x.shape[-2], x.shape[-1]) == (self.img_resolution, self.img_resolution)
            assert (rgb.shape[-2], rgb.shape[-1]) == (self.img_resolution, self.img_resolution)
        return rgb

    def sample_z(self, batch_size):
        return torch.randn((batch_size, self.z_dim), dtype=torch.float32)
    


gen = Generator(512, 256)
z = gen.sample_z(2)
x = gen(z, last_block=0)
print()