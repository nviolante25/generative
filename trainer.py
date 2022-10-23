import os
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path


class Trainer:
    def __init__(self, model, dataloader, total_steps, log_every, dest, run_name):
        self.model = model
        self.dataloader = dataloader
        self.total_steps = total_steps
        self.outdir = self.create_output_folder(dest, self.model.name, run_name)
        self.log_every = log_every

    def fit(self):
        self.step = 0 
        self.tick = 0
        self.model.train()
        with tqdm(initial=0, total=self.total_steps) as pbar:
            while self.step < self.total_steps:
                for real_samples in self.dataloader:
                    real_samples = real_samples.to(self.model.device)
                    loss = self.model.train_step(real_samples)

                    pbar.update(real_samples.shape[0])
                    self.step += real_samples.shape[0]

                    if self.step // self.log_every > self.tick:
                        self.tick +=1
                        display_images = self.model.sample_images(64)
                        display_images = Image.fromarray(display_images)
                        display_images.save(os.path.join(self.outdir, f"fakes_{str(self.tick).zfill(6)}.png"))
                pbar.set_description(f"Loss: {loss:.5f}")
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
