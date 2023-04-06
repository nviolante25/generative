import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import Adam

from torchvision.datasets import MNIST, CIFAR10, LSUN
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from PIL import Image
from pathlib import Path
import cv2 as cv
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, z_dim, img_channels):
        super().__init__()
        h = w = 4
        self.fc0 = EqualizedLinear(z_dim, z_dim * h * w)
        self.reshape = Rearrange("b (c h w) -> b c h w", c=z_dim, h=h, w=w)
        self.conv0 = EqualizedConv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.to_rgb = EqualizedConv2d(in_channels=z_dim, out_channels=img_channels, kernel_size=1)

    def forward(self, x):
        x = pixel_norm(F.leaky_relu(self.reshape(self.fc0(x)), 0.2))
        x = pixel_norm(F.leaky_relu(self.conv0(x), 0.2))
        return x


class GeneratorUpBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, img_channels):
        super().__init__()
        self.conv0 = EqualizedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv1 = EqualizedConv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.to_rgb = EqualizedConv2d(out_channels, out_channels=img_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = pixel_norm(F.leaky_relu(self.conv0(x), 0.2))
        x = pixel_norm(F.leaky_relu(self.conv1(x), 0.2))
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, img_resolution, img_channels):
        super().__init__()
        assert img_resolution % 2 == 0
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        

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
        self.add_module("b4x4", GeneratorInitBlock(z_dim, img_channels))
        self.num_blocks = int(math.log2(img_resolution) - 1)
        for i in range(1, self.num_blocks):
            res = idx2res(i)
            self.add_module(f"b{res}x{res}", GeneratorUpBlock(*channels_per_block[i], img_channels))

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
    def __init__(self, in_channels, hidden_channels, out_channels, img_channels):
        super().__init__()
        self.from_rgb = EqualizedConv2d(img_channels, in_channels, kernel_size=1)
        self.conv0 = EqualizedConv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv1 = EqualizedConv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect")

    def forward(self, x):
        x = F.leaky_relu(self.conv0(x), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.avg_pool2d(x, 2)
        return x


class DiscriminatorLastBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, img_channels):
        super().__init__()
        self.from_rgb = EqualizedConv2d(img_channels, in_channels, kernel_size=1)
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
    def __init__(self, z_dim, img_resolution, img_channels):
        super().__init__()
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

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
            self.add_module(f"b{res}x{res}", DiscriminatorDownBlock(*channels_per_block[i], img_channels))
        self.add_module("b4x4", DiscriminatorLastBlock(*channels_per_block[0], img_channels))

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


class Trainer():
    def __init__(
        self, 
        outdir, 
        dataset, 
        z_dim, 
        img_resolution, 
        img_channels, 
        batch_size, 
        r1_gamma,
        save_loss_every_kimg = 1,
        save_grid_every_kimg = 400,
        kimg_per_phase = 800,
        drift_weight=0.001,
    ):
        super().__init__()
        self.G = Generator(z_dim, img_resolution, img_channels).to("cuda")
        self.D = Discriminator(z_dim, img_resolution, img_channels).to("cuda")

        self.G_opt = Adam(self.G.parameters(), lr=0.002, betas=(0, 0.99), eps=1e-8)
        self.D_opt = Adam(self.D.parameters(), lr=0.002, betas=(0, 0.99), eps=1e-8)

        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.batch_size = batch_size

        self.dataloader = cycle(DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=16))

        self.logger = SummaryWriter(outdir)
        self.outdir = outdir
        self.drift_weight = drift_weight
        self.r1_gamma = r1_gamma
        self.save_loss_every_kimg = save_loss_every_kimg
        self.save_grid_every_kimg = save_grid_every_kimg
        self.kimg_per_phase = kimg_per_phase
        self.grid_size = 8
        self.grid_z = self.G.sample_z(self.grid_size * self.grid_size).to("cuda")


    def train_step(self, real_images, alpha, output_res):
        batch_size = real_images.shape[0]

        if output_res != real_images.shape[-1]:
            real_images = F.interpolate(real_images, size=output_res)

        # Discriminator step
        z = self.G.sample_z(batch_size).to("cuda")
        fake_images = self.G(z, alpha, output_res)

        assert fake_images.shape == real_images.shape
        fake_logits = self.D(fake_images, alpha)
        real_images.requires_grad_(True)
        real_logits = self.D(real_images, alpha)
        loss_D = discriminator_loss(real_logits, fake_logits)
        loss_D_drift = self.drift_weight * drift_loss(real_logits)
        loss_D_r1 = 0.5 * self.r1_gamma * r1_penalty_loss(real_logits, real_images)

        self.D_opt.zero_grad()
        (loss_D + loss_D_drift + loss_D_r1).backward()
        self.D_opt.step()

        # Generator step
        z = self.G.sample_z(batch_size).to("cuda")
        fake_images = self.G(z, alpha, output_res)
        fake_logits = self.D(fake_images, alpha)
        loss_G = generator_loss(fake_logits)

        self.G_opt.zero_grad()
        loss_G.backward()
        self.G_opt.step()

        losses = {
            "G": loss_G.item(),
            "D": loss_D.item(),
            "D_drift": loss_D_drift.item(),
            "D_r1":loss_D_r1.item(),
        }
        return losses

    def fit(self):
        max_res_log = int(math.log2(self.img_resolution))
        phase_resolutions = [2 ** i for i in range(2, max_res_log + 1) for _ in range(2)][1:]

        cur_nimg = 0
        cur_phase_tick = 0
        cur_phase = "stable"
        next_phase = "fade"
        phase_bar = tqdm(initial=0, total=int(self.kimg_per_phase * 1e3), position=1)
        cur_res = phase_resolutions[cur_phase_tick]
        phase_bar.set_description(f"phase: {cur_phase} at {cur_res}x{cur_res}")

        total_images = int(self.kimg_per_phase* (2 * self.G.num_blocks - 1) * 1e3)
        with tqdm(initial=0, total=total_images, position=0, desc="trainig") as pbar:
            while cur_nimg < total_images:
                cur_res = phase_resolutions[cur_phase_tick]
                if cur_phase == "stable":
                    alpha = 1.0
                elif cur_phase == "fade":
                    alpha = (cur_nimg - (cur_phase_tick * self.kimg_per_phase * 1.0e3)) / (self.kimg_per_phase * 1.0e3)
                    
                real_images = next(self.dataloader).to("cuda")
                losses = trainer.train_step(real_images, alpha, cur_res)

                self.report(losses, cur_res, cur_nimg)
                self.save_snapshot(cur_nimg, alpha, cur_res)

                if cur_nimg // (self.kimg_per_phase * 1e3) != cur_phase_tick:
                    cur_phase_tick += 1
                    assert cur_phase_tick == int(cur_nimg // (self.kimg_per_phase * 1e3))
                    cur_phase, next_phase = next_phase, cur_phase
                    phase_bar.reset()
                    if cur_phase == "stable":
                        phase_bar.set_description(f"phase: {cur_phase} at {cur_res}x{cur_res}")
                    elif cur_phase == "fade":
                        phase_bar.set_description(f"phase: {cur_phase} from {cur_res}x{cur_res} to {2*cur_res}x{2*cur_res}")

                cur_nimg += self.batch_size
                pbar.update(self.batch_size)
                phase_bar.update(self.batch_size)

        torch.save({"G": self.G.state_dict()}, os.path.join(self.outdir, f"progan_cats_{cur_res}x{cur_res}_final.pt"))

    def report(self, losses, cur_res, cur_nimg):
        if (cur_nimg % (self.save_loss_every_kimg * 1e3)) < self.batch_size:
            for name, value in losses.items():
                self.logger.add_scalar(f"Loss/{name}", value, cur_nimg)
            self.logger.add_scalar("Progressive Growing/resolution", cur_res, cur_nimg)

    @torch.no_grad()
    def save_snapshot(self, cur_nimg, alpha, cur_res):
        if (cur_nimg % (self.save_grid_every_kimg * 1e3)) < self.batch_size:
            grid_z = rearrange(self.grid_z, "(b1 b2) z -> b1 b2 z", b2=1)
            grid_img = torch.cat([self.G(z, alpha, cur_res) for z in grid_z])
            grid_img = rearrange(grid_img, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=self.grid_size, b2=self.grid_size)
            filename = os.path.join(self.outdir, f"fakes_{str(cur_nimg // 1000).zfill(8)}kimg.png")
            to_pil_image(grid_img.clip(-1, 1) * 0.5 + 0.5).save(filename)
            torch.save({"G": self.G.state_dict()}, os.path.join(self.outdir, f"network-{str(cur_nimg // 1000).zfill(8)}kimg.pt"))

def update_ema(G_ema, G, weight=0.999):
    for (name_ema, param_ema), (name, param) in zip(G_ema.named_parameters(), G.named_parameters()):
        assert name_ema == name
        param_ema.copy_(weight * param + (1 - weight) * param_ema)

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def generator_loss(fake_logits):
    return F.softplus(-fake_logits).mean()

def discriminator_loss(real_logits, fake_logits):
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

def r1_penalty_loss(real_logits, real_images):
    grad_D_wrt_reals = torch.autograd.grad(outputs=real_logits.sum(), inputs=real_images, create_graph=True)[0]
    grad_D_wrt_reals = rearrange(grad_D_wrt_reals, "b c h w -> b (c h w)")
    return torch.norm(grad_D_wrt_reals, p=2, dim=1).mean()

def drift_loss(real_logits):
    return (real_logits ** 2).mean()


class ImageDataset(Dataset):
    def __init__(self, source_dir):
      super().__init__()
      Image.init()
      self._transform = Compose([ToTensor(), Lambda(lambda x: 2.0 * x - 1.0)])
      self._image_paths = self._get_image_paths(source_dir)
      self._image_shape = list(self[0].shape)

    def _get_image_paths(self, source_dir):
        paths = [str(f) for f in Path(source_dir).rglob('*') if self.is_image_ext(f) and os.path.isfile(f)] 
        if not len(paths) > 0:
            raise ValueError(f"No images found in {source_dir}")
        return paths

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = Image.open(self._image_paths[idx])
        image_tensor = self._transform(image)
        return image_tensor

    @staticmethod
    def is_image_ext(filename: str):
        ext = str(filename).split('.')[-1].lower()
        return f'.{ext}' in Image.EXTENSION 


if __name__ == "__main__":
    dataset = ImageDataset("/data/nviolant/data_eg3d/afhq-mirror-256x256")
    trainer = Trainer(
        "./training-runs/progan-cats-256x256",
        dataset,
        z_dim=512, 
        img_resolution=256,
        img_channels=3,
        batch_size=16,
        r1_gamma=0.01,
        drift_weight=0.001,
        save_loss_every_kimg=5,
        save_grid_every_kimg=400,
        kimg_per_phase=800,
    )
    trainer.fit()


    # # Show some images
    # G = Generator(512, 32, 3)
    # G.load_state_dict(torch.load("training-runs/progan-cats-32x32/progan_cats_32x32.pt")["G"])
    # G.to("cuda")

    # grid_z = G.sample_z(256).to("cuda")
    # grid_img = G(grid_z)
    # grid_img = to_pil_image((rearrange(grid_img, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=16, b2=16)* 0.5 + 0.5).clip(0, 1))


    # def wgan_generator_loss(fake_logits):
    #     return -fake_logits.mean()

    # def wgan_discriminator_loss(real_logits, fake_logits):
    #     return fake_logits.mean() - real_logits.mean()

    # # ProGAN Mnist was trained
    # lr=0.002 with non-saturating loss
    # 