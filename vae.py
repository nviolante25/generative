import torch
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.distributions import Normal

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
        x = self.net(x)
        means, vars = x[:, : self.z_dim], F.softplus(x[:, self.z_dim :])
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
        z = self.net(z)
        means, vars = F.relu(z[:, : self.x_dim]), F.softplus(z[:, self.x_dim :])
        return means, vars


class GaussianVAE(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, device="cuda"):
        super().__init__()
        self.name = "vae"
        self.encoder = Encoder(x_dim, z_dim, hidden_dim).to(device)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim).to(device)
        self.p_z = Normal(torch.zeros(z_dim).to(device), torch.ones(z_dim).to(device))
        self.z_dim = z_dim
        self.optimizer = Adam(self.parameters(), lr=0.005)

    def train_step(self, real_samples):
        self.optimizer.zero_grad()
        # Encoder
        mean_enc, var_enc = self.encoder(real_samples)
        q_z_given_x = Normal(mean_enc, var_enc)

        # Decoder
        z_samples = q_z_given_x.rsample()
        mean_dec, var_dec = self.decoder(z_samples)
        p_x_given_z = Normal(mean_dec, var_dec)

        # Loss
        fake_samples = p_x_given_z.rsample()
        loss = self.aevb_loss(real_samples, fake_samples, q_z_given_x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def aevb_loss(self, real_samples, fake_samples, q_z_given_x):
        """
        Auto-Encoding Variational Bayes Loss
        MSE + KL
        """
        kl_loss = -(
            0.5 * (1 + torch.log(q_z_given_x.variance) - q_z_given_x.mean**2 - q_z_given_x.variance).sum(-1)
        ).mean()
        return F.mse_loss(fake_samples, real_samples) + kl_loss

    def sgvb_loss(self, real_samples, z_samples, p_x_given_z, q_z_given_x):
        return -torch.mean(
            p_x_given_z.log_prob(real_samples) + self.p_z.log_prob(z_samples) - q_z_given_x.log_prob(z_samples)
        )

    def forward(self, num_samples):
        z_samples = self.p_z.sample([num_samples]).to("cuda")
        mean, var = self.decoder(z_samples)
        p_x_given_z = Normal(mean, var)
        return p_x_given_z.rsample()

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
    run_name = "fray_faces"
    device = "cuda"
    batch_size = 256
    dataset = torch.Tensor(loadmat(data)["ff"].T / 255.0).to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    x_dim = 28 * 20
    z_dim = 20
    hidden_dim = 200
    vae = GaussianVAE(x_dim, z_dim, hidden_dim, device)

    total_epochs = 50
    trainer = Trainer(vae, dataloader, total_epochs, dest, run_name)
    trainer.fit()