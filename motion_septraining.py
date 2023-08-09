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
from multiprocessing import Process

# Get the current time
current_time = datetime.now()
# Format the time as a string
time_str = current_time.strftime("%Y%m%d_%H:%M:%S")

class MotionDataset(Dataset):
    def __init__(self, motion_path = './motion_data/run1', side_length=256, sequence_range=[5,20], skip = 4, process_id = 0):

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
        self.coords = x
        
        self.rgbs = []
        for img in self.images:
            image = Image.open(img)
            image = image.convert("RGB")
            # plt.imshow(np.array(image))
            # plt.show()
            rgb = self.transform(image).reshape(3,-1).transpose(1,0)
            self.rgbs.append(rgb)
        self.rgbs = self.rgbs[process_id]
        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return  self.coords, self.rgbs

class HDRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 1, side_length=256, motion_path = '', sequence_range=[0,1], skip = 1, process_id = 0):
        super().__init__()
        self.batch_size = batch_size
        self.side_length = side_length
        self.motion_path = motion_path
        self.sequence_range = sequence_range
        self.skip = skip
        self.process_id = process_id
        return
    
    def setup(self, stage: Any = None):
        self.dataset = MotionDataset(side_length=self.side_length, motion_path=self.motion_path, sequence_range=self.sequence_range, skip=self.skip, process_id=self.process_id)
        return
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, shuffle=False)
    
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

        out_y = out
        for i in range(1, len(self.filters)):
            out_y = self.filters[i](x) * self.linear[i - 1](out_y)
        out_y = self.output_linear(out_y)

        if self.output_act:
            out_y = torch.sin(out_y)

        return out_y


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        self.spatial = nn.Linear(in_features, out_features)
        self.spatial.weight.data *= weight_scale  # gamma
        self.spatial.bias.data.uniform_(-np.pi, np.pi)
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
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        out_y= self.model(x)
    
        loss_y = ((out_y - y)**2).mean()

        psnr2 = -10*torch.log10(loss_y)
        self.log('train_psnr', psnr2, prog_bar=True)

        return loss_y
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y_pred= self.model(x)
    
        loss_y = ((y_pred - y)**2).mean()
        y_pred = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
        y = y_pred
        y = y.detach().cpu().numpy().reshape(256, 256 ,3)
        plt.imshow(y)
        plt.axis('off')
        plt.show()

        psnr2 = -10*torch.log10(loss_y)
        print('test_psnr', psnr2)
        return psnr2
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 600], gamma=0.5)
        return [optimizer], [scheduler]

def multiple_fit(process_id, kwargs):
    print('start process {}'.format(process_id))
    logger = pl.loggers.CSVLogger(save_dir = kwargs['project_name'], name = 'log', version = time_str+'p{}'.format(process_id))
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(kwargs['project_name'],'checkpoints'),save_last=False, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100, filename=time_str+'p{}'.format(process_id))
    model = PLFourierNet()
    data_module = HDRDataModule(batch_size = kwargs['batch_size'], side_length=kwargs['side_length'], motion_path = kwargs['motion_path'], sequence_range=kwargs['sequence_range'], skip = kwargs['skip'], process_id=process_id)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=300, devices = 1, log_every_n_steps=1, enable_progress_bar=False, callbacks=[checkpoint_callback],logger=logger)
    trainer.fit(model, data_module)

if __name__ == "__main__":
    kwargs = {
        'project_name': 'motion_sep_training',
        'batch_size': 1,
        'side_length': 256,
        'pos_embed': False,
        'sequence_range': [15,175],
        'skip': 10,
        'motion_path': './motion_data/man'
    }
    num_processes = len(np.arange(kwargs['sequence_range'][0], kwargs['sequence_range'][1], kwargs['skip']))
    # num_processes = 2
    processes = []
    for i in range(num_processes):
        p = Process(target=multiple_fit, args=(i, kwargs))
        p.start()
        processes.append(p)
    
    for process in processes:
        process.join()
    print('all works done')
    # logger = pl.loggers.CSVLogger(save_dir = kwargs['project_name'], name = 'log', version = time_str)
    # progress_bar_callback = TQDMProgressBar(refresh_rate=100)
    # checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(kwargs['project_name'],'checkpoints'),save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100, filename=time_str)
    # model = PLFourierNet()
    # # ckpt_path = "motion_phase/last-v6.ckpt"
    # # for k,v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
    # #     model.state_dict()[k].copy_(v)
    # loader = HDRDataModule(batch_size = kwargs['batch_size'], side_length=kwargs['side_length'], motion_path = kwargs['motion_path'], sequence_range=kwargs['sequence_range'], skip = kwargs['skip'])
    # trainer = pl.Trainer( max_epochs=1000, devices = 1, log_every_n_steps=100, enable_progress_bar=True, callbacks=[checkpoint_callback],logger=logger)
    # trainer.fit(model, loader)