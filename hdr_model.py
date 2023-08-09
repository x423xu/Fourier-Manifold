from typing import Any
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint



class HDRDataset(Dataset):
    def __init__(self, side_length=256):

        self.images = [
            "hdr_data/r_0_0.png",
            "hdr_data/r_0_1.png",
            "hdr_data/r_0_2.png",
            "hdr_data/r_0_3.png",
            "hdr_data/r_0_4.png",
        ]
        
        self.side_length = side_length

        self.transform = Compose(
            [
                Resize(side_length),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )

        self.coords = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-1.0, 1.0, side_length),
                ]
            ),
            dim=-1,
        ).view(-1, 2)
        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.convert("L")
        return self.coords, self.transform(image).reshape(-1, 1), np.array([idx])

class HDRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, side_length=256):
        super().__init__()
        self.batch_size = batch_size
        self.side_length = side_length
        return
    
    def setup(self, stage: Any = None):
        self.dataset = HDRDataset(side_length=self.side_length)
        return
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True)

class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x, z):
        if self.pos_embed:
            x = self.pos_embedder_coord(x)
            z = self.pos_embedder_latent(z)
            out = self.filters[0](x)
            z_0 = self.modulator[0](z)
            z_0 = z_0.permute(1,0,2)
            out_z = out*z_0
            out_y = out
            for i in range(1, len(self.filters)):
                out_z = self.modulator[i](z).permute(1,0,2)+self.filters[i](x) * self.linear[i - 1](out_z)
            for i in range(1, len(self.filters)):
                out_y = self.filters[i](x) * self.linear[i - 1](out_y)
        else:
            out = self.filters[0](x)
            z_0 = self.modulator[0](z)
            z_0 = z_0.unsqueeze(1)
            out_z = out*z_0
            out_y = out
            for i in range(1, len(self.filters)):
                out_z = self.modulator[i](z).unsqueeze(1)+self.filters[i](x) * self.linear[i - 1](out_z)
            for i in range(1, len(self.filters)):
                out_y = self.filters[i](x) * self.linear[i - 1](out_y)
        out_z = self.output_linear(out_z)
        out_y = self.output_linear(out_y)

        if self.output_act:
            out_z = torch.sin(out_z)
            out_y = torch.sin(out_y)

        return out_y, out_z


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))
   
class PosEmbed(nn.Module):
    def __init__(self, magnitude = 1.0) -> None:
        super().__init__()
        self.p_fn = [torch.sin, torch.cos]
        self.freq_band = 2.0 ** torch.linspace(0.0, 9.0, 10)
        self.magnitude = magnitude

    def forward(self, x):
        self.freq_band = self.freq_band.to(x.device)
        x = x.unsqueeze(-1) * self.freq_band.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = torch.cat([p(x) for p in self.p_fn], dim=-1)
        return self.magnitude*x.view(x.shape[0], x.shape[1], -1)

class FourierNet(MFNBase):
    def __init__(
        self,
        in_size,
        hidden_size,
        latent_size,
        out_size,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
        pos_embed = False,
    ):
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayer(in_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )

        self.modulator = nn.ModuleList(
            [
                FourierLayer(latent_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )
        self.pos_embed = pos_embed
        if pos_embed:
            self.pos_embedder_coord = PosEmbed(magnitude=1.0)
            self.pos_embedder_latent = PosEmbed(magnitude=0.1)

class PLFourierNet(pl.LightningModule):
    def __init__(self, pos_embed = True):
        super(PLFourierNet, self).__init__()
        if pos_embed:
            embedding_size = 20
        else:
            embedding_size = 1
        self.model = FourierNet(
            in_size=2*embedding_size,
            hidden_size=256,
            latent_size=1*embedding_size,
            out_size=1,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
            pos_embed = pos_embed
        )
        for m in self.model.filters.parameters():
            m.requires_grad = False
    

    def forward(self, x, z):
        return self.model(x, z)
    
    def training_step(self, batch, batch_idx):
        x, y, z = batch
        z = z*0.01
        x = x.float()
        z = z.float()
        out_y, out_z = self.model(x, z)
        loss_y = ((out_y - y.mean(0))**2).mean()
        loss_z = ((out_z - y) ** 2).mean()
        psnr = -10*torch.log10(loss_z)
        self.log('train_psnr', psnr, prog_bar=True)
        return loss_y + loss_z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000, 5000], gamma=0.5)
        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    progress_bar_callback = TQDMProgressBar(refresh_rate=100)
    checkpoint_callback = ModelCheckpoint(dirpath='hdr_model',save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100)
    model = PLFourierNet(pos_embed=False)
    loader = HDRDataModule(batch_size=5, side_length=256)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=6000, devices = 1, log_every_n_steps=100, enable_progress_bar=True, callbacks=[checkpoint_callback])
    trainer.fit(model, loader)