from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
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
from datetime import datetime
import imageio

# Get the current time
current_time = datetime.now()
# Format the time as a string
time_str = current_time.strftime("%Y%m%d_%H:%M:%S")

class MotionDataset(Dataset):
    def __init__(self, motion_path = './apple', side_length=256, sequence_range=[5,20], skip = 1):

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

        x = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-1.0, 1.0, side_length),
                ]
            ),
            dim=-1,
        ).view(-1,2)
        self.coords = []
        for f in range(frames):
            # f+=0.1
            z1 = torch.sin(torch.linspace(-1.0, 1.0, side_length)*(2**(10))*np.pi+f/20*2*np.pi).view(-1,1)
            z2 = torch.cos(torch.linspace(-1.0, 1.0, side_length)*(2**(10))*np.pi+f/20*2*np.pi).view(1,-1)
            z = z1+z2
            z = z.view(-1,1)
            self.coords.append(torch.cat([x,z], dim=-1))
        
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
        return  self.coords[idx], self.rgbs[idx]

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
    
    def test_dataloader(self):
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
        out = self.filters[0](x)
        z_0 = self.amplitude_modulator[0](z)
        out_z = out+z_0
        out_y = out
        for i in range(1, len(self.filters)):
            out_z = self.amplitude_modulator[i](z)+self.filters[i](x) * self.linear[i - 1](out_z)
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

        # self.spatial = nn.Linear(in_features, out_features)
        # self.spatial.weight.data *= weight_scale  # gamma
        # self.spatial.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x, z=None):
        if z is not None:f = torch.sin(z*self.linear(x))
        else: f = torch.sin(self.linear(x))
        # s = torch.exp(-self.spatial(x)**2)
        # return f*s
        return f

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

        # self.phase_modulator = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(latent_size, 2),
        #         )
        #         for _ in range(n_layers + 1)
        #     ]
        # )

        self.amplitude_modulator = nn.ModuleList(
            [
                FourierLayer(latent_size, hidden_size, input_scale / np.sqrt(n_layers + 1))
                for _ in range(n_layers + 1)
            ]
        )

        # self.frequency_modulator = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(latent_size, hidden_size),
        #         )
        #         for _ in range(n_layers + 1)
        #     ]
        # )

class PLFourierNet(pl.LightningModule):
    def __init__(self):
        super(PLFourierNet, self).__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()
        self.model = FourierNet(
            in_size=2,
            hidden_size=256,
            latent_size=1,
            out_size=3,
            n_layers=3,
            input_scale=256,
            weight_scale=1,
        )
    
    def forward(self, x, z):
        return self.model(x, z)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = x[...,2:]
        x = x[...,:2]
        z=z
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

        return loss_z
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = x[...,2:]
        x = x[...,:2]
        # z = 1/2*z+1/2*torch.roll(z, 1, dims=0)
        # y = torch.roll(y, 1, dims=0)
        x = x.float()
        z = z.float()
        out_y, out_z = self.model(x, z)
        out_z = out_z.squeeze()
        y = y.squeeze()
        loss_z = ((out_z - y) ** 2).mean()

        psnr = -10*torch.log10(loss_z)
        self.log('test_psnr', psnr, prog_bar=True)
        y_pred = out_z

        ys = []
        for i,y in enumerate(y_pred):
            y = (y-y.min())/(y.max()-y.min())*255
            y = y.int()
            y = y.detach().cpu().numpy().reshape(256, 256,3)
            y = y.astype(np.uint8)
            ys.append(y)
            # plt.imshow(y)
            # plt.show()
            if not os.path.exists('output/{}'.format(os.path.basename(__file__).rstrip('.py'))):
                os.makedirs('output/{}'.format(os.path.basename(__file__).rstrip('.py')))
            imageio.imwrite('output/{}/frame_{}.png'.format(os.path.basename(__file__).rstrip('.py'),i), y)
        imageio.mimsave(f'test_relightning.gif', ys, duration = 300, loop=0)

        return psnr
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 600])
        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    kwargs = {
        'project_name': 'relightning_amplitude',
        'batch_size': 10,
        'side_length': 256,
        'pos_embed': False,
        'sequence_range': [5,20],
        'skip': 2,
        'motion_path': './apple'
    }
    logger = pl.loggers.CSVLogger(save_dir = kwargs['project_name'], name = 'log', version = time_str)
    progress_bar_callback = TQDMProgressBar(refresh_rate=100)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(kwargs['project_name'],'checkpoints'),save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100, filename=time_str)
    model = PLFourierNet()
    ckpt_path = "/home/xxy/Documents/code/Fourier-Manifold/relightning_amplitude/checkpoints/20230809_01:22:56.ckpt"
    # for k,v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    #     model.state_dict()[k].copy_(v)
    loader = HDRDataModule(batch_size = kwargs['batch_size'], side_length=kwargs['side_length'], motion_path = kwargs['motion_path'], sequence_range=kwargs['sequence_range'], skip = kwargs['skip'])
    trainer = pl.Trainer(accelerator='gpu',max_epochs=1000, devices = 2, log_every_n_steps=100, enable_progress_bar=True, callbacks=[checkpoint_callback],logger=logger)
    trainer.fit(model, loader)
    # model = model.load_from_checkpoint(ckpt_path)
    trainer.test(model, loader)