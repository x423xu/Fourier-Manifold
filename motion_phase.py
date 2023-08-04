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
from glob import glob
import os
import matplotlib.pyplot as plt
from natsort import natsorted

class MotionDataset(Dataset):
    def __init__(self, motion_path = './motion_data/run1', side_length=256, sequence_range=[5,20], skip = 4):

        self.images = glob(os.path.join(motion_path, '*.png'))
        self.images = natsorted(self.images)
        
        self.side_length = side_length

        self.transform = Compose(
            [
                Resize([side_length, side_length]),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )
        self.images = self.images[sequence_range[0]:sequence_range[1]:skip]
        frames = len(np.arange(sequence_range[0], sequence_range[1], skip))

        self.coords = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-1.0, 1.0, side_length),
                ]
            ),
            dim=-1,
        ).view(-1,2)
        self.rgbs = []
        for img in self.images:
            image = Image.open(img)
            image = image.convert("RGB")
            # plt.imshow(np.array(image))
            # plt.show()
            rgb = self.transform(image).reshape(3,-1).transpose(1,0)
            self.rgbs.append(rgb)

        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.coords, self.rgbs[idx], np.array([idx])

class HDRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 1, side_length=256, motion_path = '', sequence_range=[0,1], skip = 1):
        super().__init__()
        self.batch_size = batch_size
        self.side_length = side_length
        self.motion_path = motion_path
        self.sequence_range = sequence_range
        self.skip = skip
        return
    
    def setup(self, stage: Any = None):
        self.dataset = MotionDataset(side_length=self.side_length, motion_path=self.motion_path, sequence_range=self.sequence_range, skip=self.skip)
        return
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=8, shuffle=False)
    
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
            out = self.filters[0](x+self.phase_modulator[0](z).unsqueeze(1))
            z_0 = self.amplitude_modulator[0](z).unsqueeze(1)
            out_z = out*z_0
            out_y = out
            for i in range(1, len(self.filters)):
                out_z = self.amplitude_modulator[i](z).unsqueeze(1)+self.filters[i](x+self.phase_modulator[i](z).unsqueeze(1)) * self.linear[i - 1](out_z)
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
        pos_embed=False,
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

        self.phase_modulator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_size, 1),
                )
                for _ in range(n_layers + 1)
            ]
        )

        self.amplitude_modulator = nn.ModuleList(
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
    def __init__(self, pos_embed = False):
        super(PLFourierNet, self).__init__()
        # self.automatic_optimization = False
        if pos_embed:
            embedding_size = 20
        else:
            embedding_size = 1
        self.model = FourierNet(
            in_size=2*embedding_size,
            hidden_size=256,
            latent_size=1*embedding_size,
            out_size=3,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
            pos_embed = pos_embed
        )
    
    def _freeze_backbone(self):
        for p in self.model.filters.parameters():
            p.requires_grad = False
        for p in self.model.linear.parameters():
            p.requires_grad = False
        for p in self.model.output_linear.parameters():
            p.requires_grad = False

    def forward(self, x, z):
        return self.model(x, z)
    
    def training_step(self, batch, batch_idx):
        x, y, z = batch
        z = z*0.01
        x = x.float()
        z = z.float()
        out_y, out_z = self.model(x, z)
    
        loss_y = ((out_y - y[0])**2).mean()

        # self._freeze_backbone()
        loss_z = ((out_z - y) ** 2).mean()


        psnr = -10*torch.log10(loss_z)
        psnr2 = -10*torch.log10(loss_y)
        self.log('train_psnr', psnr, prog_bar=True)
        self.log('backbone_psnr', psnr2, prog_bar=True)

        return loss_z + loss_y
    
    def configure_optimizers(self):
        # mod_optimizer = torch.optim.Adam(self.model.modulator.parameters(), lr=1e-2)
        # filter_optimizer = torch.optim.Adam(self.model.filters.parameters(), lr=1e-2)
        # mod_scheduler = torch.optim.lr_scheduler.MultiStepLR(mod_optimizer, milestones=[200, 400, 600, 800, 1000], gamma=0.5)
        # filter_scheduler = torch.optim.lr_scheduler.MultiStepLR(filter_optimizer, milestones=[200, 400, 600, 800, 1000], gamma=0.5)
        # return [mod_optimizer, filter_optimizer], [mod_scheduler, filter_scheduler]

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000, 5000], gamma=0.5)
        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    kwargs = {
        'project_name': 'motion_phase',
        'batch_size': 10,
        'side_length': 256,
        'pos_embed': False,
        'sequence_range': [15,75],
        'skip': 10,
        'motion_path': './motion_data/man'
    }
    progress_bar_callback = TQDMProgressBar(refresh_rate=100)
    checkpoint_callback = ModelCheckpoint(dirpath=kwargs['project_name'],save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100)
    model = PLFourierNet(kwargs['pos_embed'])
    # ckpt_path = "motion_phase/last-v6.ckpt"
    # for k,v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    #     model.state_dict()[k].copy_(v)
    loader = HDRDataModule(batch_size = kwargs['batch_size'], side_length=kwargs['side_length'], motion_path = kwargs['motion_path'], sequence_range=kwargs['sequence_range'], skip = kwargs['skip'])
    trainer = pl.Trainer( max_epochs=1000, devices = 1, log_every_n_steps=100, enable_progress_bar=True, callbacks=[checkpoint_callback],logger=pl.loggers.CSVLogger(save_dir=kwargs['project_name'], name='log'))
    trainer.fit(model, loader)