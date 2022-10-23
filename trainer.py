import os
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path


class Trainer:
    def __init__(self, model, dataloader, total_epochs, dest, run_name):
        self.model = model
        self.dataloader = dataloader
        self.total_epochs = total_epochs
        self.outdir = self.create_output_folder(dest, self.model.name, run_name)

    def fit(self):
        with tqdm(initial=0, total=self.total_epochs) as pbar:
            for epoch in range(self.total_epochs):
                for real_samples in self.dataloader:
                    loss = self.model.train_step(real_samples)

                pbar.set_description(f"Loss: {loss:.5f}")
                pbar.update(1)
                display_images = self.model.sample_images(64)
                display_images = Image.fromarray(display_images)
                display_images.save(os.path.join(self.outdir, f"fakes_{str(epoch).zfill(6)}.png"))
        self.save()

    def save(self):
        checkpoint = {
            "model": self.model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.outdir, "model-checkpoint.pt"))

    @staticmethod
    def create_output_folder(dest, model_name, run_name):
        os.makedirs(dest, exist_ok=True)
        dataset_name = Path(run_name).name
        num = len(os.listdir(dest))
        outdir = os.path.join(dest, f"{str(num).zfill(4)}-{model_name}-{dataset_name}")
        os.makedirs(outdir)
        return outdir
