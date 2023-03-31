import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, CIFAR10, ImageNet
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from torch.optim import Adam
from einops import rearrange
from tqdm import tqdm

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_num, embedding_dim):
        super().__init__()
        self.codebook = nn.Parameter(torch.Tensor(size=(embedding_num, embedding_dim)), requires_grad=True)
        nn.init.kaiming_normal_(self.codebook)

    def forward(self, latent):
        assert len(latent.shape) == 4
        b, c, h, w = latent.shape
        latent = rearrange(latent, 'b c h w -> (b h w) c')
        indices = torch.cdist(latent, self.codebook).argmin(dim=1)
        latent_quantized = self.codebook[indices]

        codebook_loss = F.mse_loss(latent_quantized, latent.detach())
        commitment_loss = F.mse_loss(latent_quantized.detach(), latent)

        latent_quantized = latent + (latent_quantized - latent).detach()  # copy gradients from z to z_quantized
        latent_quantized = rearrange(latent_quantized, '(b h w) c -> b c h w', b=b, h=h, w=w, c=c)

        return latent_quantized, codebook_loss, commitment_loss


class VQVAEMnist(nn.Module):
    def __init__(self, codebook_dim, latent_dim) -> None:
        super().__init__()
        self.vq_layer = VectorQuantizer(codebook_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=latent_dim, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
        )

    def train_step(self, x):
        codebook_loss, commitment_loss, recontruction = self.reconstruct(x)

        reconstruction_loss = F.mse_loss(x, recontruction)
        total_loss = reconstruction_loss + codebook_loss + 0.25 * commitment_loss
        return total_loss

    def reconstruct(self, x):
        latent = self.encoder(x)
        latent_quantized, codebook_loss, commitment_loss = self.vq_layer(latent)
        recontruction = self.decoder(latent_quantized)
        return codebook_loss, commitment_loss, recontruction

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1),
        ) 

    def forward(self, x):
        return F.relu(self.block(x) +  x)
        
class VQVAE(nn.Module):
    def __init__(self, codebook_dim, latent_dim) -> None:
        super().__init__()
        self.vq_layer = VectorQuantizer(codebook_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
        )
        self.decoder = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
            nn.Upsample(scale_factor=2.0, mode="bilinear"),
            nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0, mode="bilinear"),
            nn.Conv2d(in_channels=latent_dim, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def train_step(self, x):
        codebook_loss, commitment_loss, recontruction = self.reconstruct(x)

        reconstruction_loss = F.mse_loss(x, recontruction)
        total_loss = reconstruction_loss + codebook_loss + 0.25 * commitment_loss
        return total_loss

    def reconstruct(self, x):
        latent = self.encoder(x)
        latent_quantized, codebook_loss, commitment_loss = self.vq_layer(latent)
        recontruction = self.decoder(latent_quantized)
        return codebook_loss, commitment_loss, recontruction


if __name__ == "__main__":
    dataset = CIFAR10("./data/cifar10", transform=Compose([ToTensor()]), download=True)
    batch_size=128
    dataloader = cycle(DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True))
    device = "cuda"

    vq_vae = VQVAE(512, 256).to(device)
    opt = Adam(vq_vae.parameters(), lr=2e-4)

    total_images = int(1000e3)
    cur_nimg = 0
    all_losses = []
    with tqdm(initial=0, total=total_images) as pbar:
        while cur_nimg < total_images:
            images = next(dataloader)[0].to(device)
        
            loss = vq_vae.train_step(images)

            opt.zero_grad()
            loss.backward()
            opt.step()

            cur_nimg += batch_size
            pbar.update(batch_size)
            pbar.set_description(f"Loss {loss.item():.4f}")
            all_losses.append(loss.item())

    print()
    grid = next(dataloader)[0][:64].to(device)
    _, _, rec_grid = vq_vae.reconstruct(grid)
    to_pil_image(rearrange(grid, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=8)).save("./grid.png")
