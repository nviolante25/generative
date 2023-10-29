import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, RandomHorizontalFlip, Resize, Grayscale
from torchvision.transforms.functional import to_pil_image
from diffusers import UNet2DModel
from einops import rearrange
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter


def get_model(img_channels=3):
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels, out_channels=img_channels, in_channels=img_channels, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)


class TrainerIADB:
    def __init__(self, outdir, total_images, dataloader, img_channels=3) -> None:
        self.model = get_model(img_channels).cuda()
        self.model_ema = deepcopy(self.model).requires_grad_(False).cuda()
        self.opt = AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))

        self.outdir = outdir
        self.logger = SummaryWriter(self.outdir)
        os.makedirs(outdir, exist_ok=True)

        self.total_images = total_images
        self.dataloader = dataloader
        self.save_grid_every_kimg = 1000
        self.img_channels = img_channels
        self.grid_x0 = torch.randn(16, self.img_channels, 32, 32, device=self.model.device)

    def fit(self, batch_size=256, img_resolution=32):
        cur_nimg = 0
        pbar = tqdm(initial=cur_nimg, total=self.total_images, position=0, miniters=10)

        x_source = torch.zeros((batch_size, self.img_channels, img_resolution, img_resolution), device=self.model.device)
        x_target = torch.zeros((batch_size, self.img_channels, img_resolution, img_resolution), device=self.model.device)
        blend    = torch.zeros((batch_size, 1, 1, 1), device=self.model.device)

        cur_warmup = 0
        total_warmup = 10

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        while cur_nimg < self.total_images:
            x_target_ = next(self.dataloader)  
            x_source_ = torch.randn_like(x_target) 
            blend_ = torch.rand((batch_size, 1, 1, 1))

            x_target.copy_(x_target_)
            x_source.copy_(x_source_)
            blend.copy_(blend_)

            # Warmup
            if cur_warmup < total_warmup:
                with torch.cuda.stream(s):
                    x_blend = x_source + blend * (x_target - x_source)
                    pred = self.model(x_blend, blend[:, 0, 0, 0])["sample"]
                    loss = F.mse_loss(pred, x_target - x_source)
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
                    self.update_ema()
                cur_warmup += 1

            # Capture
            elif cur_warmup == total_warmup:
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    x_blend = x_source + blend * (x_target - x_source)
                    pred = self.model(x_blend, blend[:, 0, 0, 0])["sample"]
                    loss = F.mse_loss(pred, x_target - x_source)
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
                    self.update_ema()
                cur_warmup += 1

            # Replay
            else:
                g.replay()

            losses = {"loss": loss.item()}
            self.report(losses, cur_nimg)
            if (cur_nimg % (self.save_grid_every_kimg * 1e3)) < batch_size:
                self.save_snapshot(cur_nimg)

            cur_nimg += batch_size
            pbar.update(batch_size)

    def update_ema(self, beta = 0.999):
        src_params = dict(self.model.named_parameters())
        for name_ema, param_ema in self.model_ema.named_parameters():
            param_ema.data.copy_(beta * param_ema.data + (1.0 - beta) * src_params[name_ema].data)

    def report(self, losses, cur_nimg):
        for name, value in losses.items():
            self.logger.add_scalar(f"Loss/{name}", value, cur_nimg)
    
    @torch.no_grad()
    def save_snapshot(self, cur_nimg):
        # Grid of images
        grid_img = sample_iadb(self.model_ema, self.grid_x0)
        filename = os.path.join(self.outdir, f"fakes_{str(cur_nimg // 1000).zfill(8)}kimg.png")
        grid_img = rearrange(grid_img, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=4, b2=4)
        to_pil_image(grid_img.clip(-1, 1) * 0.5 + 0.5).save(filename)
        
        # Weights and Optimizer
        checkpoint = {
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "opt": self.opt.state_dict(),
            "cur_nimg": cur_nimg,
        }
        torch.save(checkpoint, os.path.join(self.outdir, f"network-{str(cur_nimg // 1000).zfill(8)}kimg.pt"))

@torch.no_grad()
def sample_iadb(model, x_source, num_steps=128):
    x_t = x_source
    for t in range(num_steps):
        blend_t = torch.FloatTensor([t / num_steps]).to(model.device)
        blend_tp1 = torch.FloatTensor([(t + 1.0) / num_steps]).to(model.device)
        x_t = x_t + (blend_tp1 - blend_t) * model(x_t, blend_t)["sample"]
    x_target = x_t
    return x_target

import math

@torch.no_grad()
def sample_iadb_2nd_order(model, x_source, num_steps=128):
    x_t = x_source
    def cos_schedule(t, num_steps):
        t = torch.FloatTensor([t / num_steps])
        return 1.0 - torch.cos(0.5 * torch.pi * t)
    
    for t in range(num_steps):
        blend_t = cos_schedule(t, num_steps).cuda()
        blend_mid = cos_schedule(t + 0.5, num_steps).cuda()
        blend_tp1 = cos_schedule(t + 1.0, num_steps).cuda()
        x_mid = x_t + (blend_mid - blend_t) * model(x_t, blend_t)["sample"]
        x_t = x_t + (blend_tp1 - blend_t) * model(x_mid, blend_mid)["sample"]
    x_target = x_t
    return x_target


def cycle(dataloader):
    while True:
        for data in dataloader:
            if type(data) == list:
                yield data[0]
            else:
                yield data
            

if __name__ == "__main__":
    dataset = "cifar10"

    if dataset == "celeb":
        transform = Compose([ToTensor(), Resize(32), RandomHorizontalFlip(), Lambda(lambda x: 2.0 * x - 1.0)])
        dataset = ImageFolder("./data/celeba_50k", transform=transform)
        outdir = "./training-runs-iadb/celeba-32x32-ema"
        img_channels=3

    elif dataset == "cifar10":
        transform = Compose([ToTensor(), RandomHorizontalFlip(), Lambda(lambda x: 2.0 * x - 1.0)])
        dataset = CIFAR10("./data/cifar10", transform=transform)
        outdir = "./training-runs-iadb/cifar10-32x32-ema"
        img_channels=3

    elif dataset == "mnist":
        transform = Compose([Grayscale(), Resize(32), ToTensor(), Lambda(lambda x: 2.0 * x - 1.0)])
        dataset = MNIST("./data/mnist", transform=transform)
        outdir = "./training-runs-iadb/mnist-32x32-ema"
        img_channels=1

    batch_size=200
    dataloader = cycle(DataLoader(dataset, batch_size=batch_size, num_workers=8, persistent_workers=True, prefetch_factor=4, drop_last=True, shuffle=True))
    total_images = int(20e6)

    trainer = TrainerIADB(outdir, total_images, dataloader, img_channels)
    trainer.fit(batch_size=batch_size)

    # # Generate grid
    # checkpoint = "./training-runs-iadb/celeba-32x32-ema/network-00002000kimg.pt"
    # checkpoint = "./training-runs-iadb/cifar10-32x32-ema/network-00019000kimg.pt"
    # checkpoint = "./training-runs-iadb/mnist-32x32-ema/network-00004000kimg.pt"
    # img_channels = 1
    # model_ema = get_model(img_channels).cuda()
    # model_ema.load_state_dict(torch.load(checkpoint)["model_ema"])
    # grid_x0 = torch.randn(64, img_channels, 32, 32, device=model_ema.device)
    # x1 = sample_iadb(model_ema, grid_x0)

    # filename = "./iadb_grid_mnist.png"
    # grid_img = rearrange(x1, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=8, b2=8)
    # to_pil_image(grid_img.clip(-1, 1) * 0.5 + 0.5).save(filename)
