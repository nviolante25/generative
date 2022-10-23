import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.distributions import Normal, Independent

from trainer import Trainer


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        z = self.net(x)
        means, vars = z[:, : self.z_dim], F.softplus(z[:, self.z_dim :])
        return means, vars


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * x_dim),
        )
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

    def forward(self, z):
        x = self.net(z)
        means, vars = F.sigmoid(x[:, : self.x_dim]), F.softplus(x[:, self.x_dim :])
        return means, vars


class GaussianVAE(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, device="cuda"):
        super().__init__()
        self.name = "vae"
        self.encoder = Encoder(x_dim, z_dim, hidden_dim).to(device)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim).to(device)
        self.p_z = Normal(torch.zeros(z_dim).to(device), torch.ones(z_dim).to(device))
        self.z_dim = z_dim
        self.optimizer = Adam(self.parameters(), lr=0.0005)
        self.device=device

    def train_step(self, real_samples):
        self.optimizer.zero_grad()
        # Encoder
        mean_enc, var_enc = self.encoder(real_samples)
        q_z_given_x = Independent(Normal(mean_enc, var_enc), -1)

        # Decoder
        z_samples = q_z_given_x.rsample()
        mean_dec, var_dec = self.decoder(z_samples)
        p_x_given_z = Independent(Normal(mean_dec, var_dec), -1)

        # Loss
        loss = self.aevb_loss(real_samples, q_z_given_x, p_x_given_z)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def aevb_loss(self, real_samples, q_z_given_x, p_x_given_z):
        kl_loss = -(
            0.5 * (1 + torch.log(q_z_given_x.variance) - q_z_given_x.mean**2 - q_z_given_x.variance).sum(-1)
        ).mean()
        reconstruction_loss = - p_x_given_z.log_prob(real_samples).mean()
        return reconstruction_loss + kl_loss

    def forward(self, num_samples):
        z_samples = self.p_z.sample([num_samples]).to("cuda")
        mean, _ = self.decoder(z_samples)
        return mean

    torch.no_grad()
    def sample_images(self, num_samples=16):
        assert num_samples == int(np.sqrt(num_samples)) ** 2
        display_images = self(num_samples).view(num_samples, 1, 28, 20)

        display_images = make_grid(display_images, nrow=int(np.sqrt(num_samples)))
        display_images = (
            (255 * torch.clip(display_images, 0, 1)).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        )
        return display_images


def sample_2d_manifold():
    z = torch.linspace(-3, 3, 16)
    zz1, zz2 = torch.meshgrid(z, z, indexing="xy")
    return torch.cat([zz1[None, ...], zz2[None, ...]]).view(-1, 2)


if __name__ == "__main__":
    data = "./data/frey_rawface.mat"
    dest = "./training-runs"
    run_name = "frey_faces"
    device = "cuda"
    batch_size = 100
    dataset = torch.Tensor(loadmat(data)["ff"].T / 255.0).to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    x_dim = 28 * 20
    z_dim = 20
    hidden_dim = 200
    vae = GaussianVAE(x_dim, z_dim, hidden_dim, device)

    total_steps = int(1e7)
    log_every = int(5e5)
    trainer = Trainer(vae, dataloader, total_steps, log_every, dest, run_name)
    trainer.fit()
