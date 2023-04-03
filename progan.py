import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import Adam

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from tqdm import tqdm


def idx2res(idx):
    return int(2 ** (idx + 2))

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
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
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
            [z_dim, z_dim, z_dim],                      # 4x4
            [z_dim, z_dim, z_dim],                      # 8x8
            [z_dim, z_dim, z_dim],                      # 16x16
            [z_dim, z_dim, z_dim],                      # 32x32
            [z_dim, z_dim // 2, z_dim // 2],            # 64x64
            [z_dim // 2, z_dim // 4, z_dim // 4],       # 128x128
            [z_dim // 4, z_dim // 8, z_dim // 8],       # 256x256
            [z_dim // 8, z_dim // 16, z_dim // 16],     # 512x512   
            [z_dim // 16, z_dim // 32, z_dim // 32],    # 1024X1024
        ]

        # Create named blocks: b4x4, b8x8,...,b1024x1024
        self.add_module("b4x4", GeneratorInitBlock(z_dim))
        self.num_blocks = int(math.log2(img_resolution) - 1)
        for i in range(1, self.num_blocks):
            res = idx2res(i)
            self.add_module(f"b{res}x{res}", GeneratorUpBlock(*channels_per_block[i]))

    def forward(self, z, alpha=1.0, output_res=None):
        """
        Input:
            z: [batch_size, z_dim]
            alpha: default no faded transition

        Output:
            x: [batch_size, 3, img_resolution, img_resolution]
        """
        assert z.shape[1] == self.z_dim
        if output_res is None:
            output_res = self.img_resolution

        x = z
        for res in [2 ** i for i in range(2, 1 + int(math.log2(output_res)))]: # 4, 8, 16, etc
            block = self.get_submodule(f"b{res}x{res}")
            x_prev = x
            x = block(x)
        to_rgb = self.get_submodule(f"b{res}x{res}.to_rgb")
        rgb = to_rgb(x)
        rgb = self.fade(rgb, x_prev, alpha)
        return rgb

    def sample_z(self, batch_size):
        return torch.randn((batch_size, self.z_dim), dtype=torch.float32)

    def fade(self, rgb, x_prev, alpha):
        previous_res = rgb.shape[-1] // 2
        if alpha < 1.0:
            # Use to_rgb of lower resolution block
            to_rgb_prev = self.get_submodule(f"b{previous_res}x{previous_res}.to_rgb")
            x_prev = F.interpolate(x_prev, scale_factor=2.0, mode="nearest")
            return alpha * rgb + (1.0 - alpha) * to_rgb_prev(x_prev)
        return rgb


class DiscriminatorDownBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.from_rgb = EqualizedConv2d(3, in_channels, kernel_size=1)
        self.conv0 = EqualizedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv1 = EqualizedConv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = F.leaky_relu(self.conv0(x), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.avg_pool2d(x, 2)
        return x


class DiscriminatorLastBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.from_rgb = EqualizedConv2d(3, in_channels, kernel_size=1)
        self.conv0 = EqualizedConv2d(in_channels + 1, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv1 = EqualizedConv2d(hidden_channels, out_channels, kernel_size=4)
        self.to_logits = EqualizedLinear(out_channels, 1)

    def forward(self, x):
        x = self.mini_batch_std_dev(x)
        x = F.leaky_relu(self.conv0(x), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = rearrange(x, "b c 1 1 -> b c")
        logits = self.to_logits(x)
        return logits

    @staticmethod
    def mini_batch_std_dev(x):
        b, _, h, w = x.shape
        mini_batch_std = repeat(x.std(dim=0).mean(), " -> b 1 h w", b=b, h=h, w=w)
        return torch.cat([x, mini_batch_std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, z_dim, img_resolution):
        super().__init__()
        self.z_dim = z_dim
        self.img_resolution = img_resolution

        channels_per_block = [
            [z_dim, z_dim, z_dim],                      # 4x4
            [z_dim, z_dim, z_dim],                      # 8x8
            [z_dim, z_dim, z_dim],                      # 16x16
            [z_dim, z_dim, z_dim],                      # 32x32
            [z_dim // 2, z_dim, z_dim],                 # 64x64
            [z_dim // 4, z_dim // 2, z_dim // 2],       # 128x128
            [z_dim // 8, z_dim // 4, z_dim // 4],       # 256x256
            [z_dim // 16, z_dim // 8, z_dim // 8],      # 512x512   
            [z_dim // 32, z_dim // 16, z_dim // 16],    # 1024X1024
        ]

        self.total_num_blocks = int(math.log2(img_resolution) - 1)
        for i in reversed(range(1, self.total_num_blocks)):
            res = idx2res(i)
            self.add_module(f"b{res}x{res}", DiscriminatorDownBlock(*channels_per_block[i]))
        self.add_module("b4x4", DiscriminatorLastBlock(*channels_per_block[0]))

    def forward(self, rgb, alpha=1.0):

        input_res = rgb.shape[-1]
        block = self.get_submodule(f"b{input_res}x{input_res}")
        x = block.from_rgb(rgb)
        x = block(x)
        x = self.fade(rgb, x, alpha)

        for res in reversed([2 ** i for i in range(2, int(math.log2(input_res)))]):
            block = self.get_submodule(f"b{res}x{res}")
            x = block(x)
        
        return x

    def fade(self, rgb, x, alpha):
        input_res = rgb.shape[-1]
        previous_res = input_res // 2
        if alpha < 1.0 and input_res < self.img_resolution:
            # Use from_rgb of previous block
            from_rgb_prev = self.get_submodule(f"b{previous_res}x{previous_res}.from_rgb")
            x_prev = from_rgb_prev(F.avg_pool2d(rgb, 2))
            return alpha * x + (1 - alpha) * x_prev
        return x


class Trainer(nn.Module):
    def __init__(self, z_dim, img_resolution) -> None:
        super().__init__()
        self.G = Generator(z_dim, img_resolution)
        self.D = Discriminator(z_dim, img_resolution)

        self.G_opt = Adam(self.G.parameters(), lr=0.001, betas=(0, 0.99), eps=1e-8)
        self.D_opt = Adam(self.D.parameters(), lr=0.001, betas=(0, 0.99), eps=1e-8)


    def train_step(self, real_images):
        batch_size = real_images.shape[0]

        # Discriminator step
        z = self.G.sample_z(batch_size).to("cuda")
        fake_images = self.G(z)
        fake_logits = self.D(fake_images)
        real_logits = self.D(real_images)
        loss_D = discriminator_loss(real_logits, fake_logits)

        self.D_opt.zero_grad()
        loss_D.backward()
        self.D_opt.step()

        # Generator step
        z = self.G.sample_z(batch_size).to("cuda")
        fake_images = self.G(z)
        fake_logits = self.D(fake_images)
        loss_G = generator_loss(fake_logits)

        self.G_opt.zero_grad()
        loss_G.backward()
        self.G_opt.step()

    def fit(total_nimg=int(25e6)):
        pass

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def generator_loss(fake_logits):
    return F.softplus(-fake_logits).mean()


def discriminator_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


if __name__ == "__main__":
    dataset = MNIST("./data/mnist", transform=Compose([ToTensor(), Resize(32), Lambda(lambda x: 2.0 * x - 1.0)]), download=True)
    trainer = Trainer(512, 32).to("cuda")

    batch_size = 16
    dataloader = cycle(DataLoader(dataset, batch_size=batch_size, drop_last=True))
    total_images = int(100e6)
    cur_nimg = 0
    with tqdm(initial=0, total=total_images) as pbar:
        while cur_nimg < total_images:
            real_images = repeat(next(dataloader)[0].to("cuda"), "b 1 h w -> b c h w", c=3)
            trainer.train_step(real_images)

            cur_nimg += batch_size
            pbar.update(batch_size)