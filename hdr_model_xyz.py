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
                    torch.linspace(0.0, 0.04, 5),
                ]
            ),
            dim=-1,
        ).view(-1,5,3).transpose(0,1)
        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images = []
        for img in self.images:
            image = Image.open(img)
            image = image.convert("RGB")
            images.append(self.transform(image).reshape(3, -1).transpose(0,1))
        return self.coords, torch.stack(images)

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

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linear[i - 1](out)
        out = self.output_linear(out)
        if self.output_act:
            out = torch.sin(out)

        return out


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
        out_size,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
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

class PLFourierNet(pl.LightningModule):
    def __init__(self):
        super(PLFourierNet, self).__init__()
        self.model = FourierNet(
            in_size=3,
            hidden_size=256,
            out_size=3,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        )
    

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        out = self.model(x)
        loss = ((out - y) ** 2).mean()
        psnr = -10*torch.log10(loss)
        self.log('train_psnr', psnr, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 3000, 4000, 5000], gamma=0.5)
        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    progress_bar_callback = TQDMProgressBar(refresh_rate=100)
    checkpoint_callback = ModelCheckpoint(dirpath='hdr_xyz',save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100)
    model = PLFourierNet()
    # ckpt_path = "/home/xxy/Documents/code/multiplicative-filter-networks/lightning_logs/version_8/checkpoints/epoch=999-step=1000.ckpt"
    # for k,v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    #     model.state_dict()[k].copy_(v)
    loader = HDRDataModule(batch_size=5, side_length=256)
    trainer = pl.Trainer(max_epochs=6000, devices = 1, log_every_n_steps=100, enable_progress_bar=True, callbacks=[checkpoint_callback])
    trainer.fit(model, loader)