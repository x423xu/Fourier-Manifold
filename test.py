import torch
import cv2
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
from motion_phase import PLFourierNet
import numpy as np

def test_motion_xyt():
    model = PLFourierNet()
    ckpt_path = "/home/xxy/Documents/code/Fourier-Manifold/motion_xyt/last-v2.ckpt"
    for k,v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
        model.state_dict()[k].copy_(v)
    side_length = 256
    sequence_range=[5,20]
    skip = 4
    frames = len(np.arange(sequence_range[0], sequence_range[1], skip))

    x = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-0.01, 0.00, frames),
                ]
            ),
            dim=-1,
        ).view(-1,frames,3).transpose(0,1).to(model.device)
    y_pred = model(x)
    print(y_pred.shape)
    y_pred = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
    fig, axs = plt.subplots(1,frames, figsize=(20,20))
    for i,y in enumerate(y_pred):
        y = y.detach().cpu().numpy().reshape(side_length, side_length,3)
        if isinstance(axs, np.ndarray):
            axs[i].imshow(y)
            axs[i].axis('off')
        else:
            axs.imshow(y)
            axs.axis('off')
    plt.show()

def test_motino_phase(z):
    ckpt_path = "motion_phase/last-v1.ckpt"
    model = PLFourierNet(pos_embed = False)
    for k,v in torch.load(ckpt_path, map_location='cpu')['state_dict'].items():
        model.state_dict()[k].copy_(v)
    side_length = 256

    x = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(-1.0, 1.0, side_length),
                    torch.linspace(-1.0, 1.0, side_length),
                ]
            ),
            dim=-1,
        ).view(-1,2)
    z = torch.tensor([[z]]).float().to(model.device)
    _, y_pred = model(x, z)
    y_pred = (y_pred-y_pred.min())/(y_pred.max()-y_pred.min())
    for y in y_pred:
        y = y.detach().cpu().numpy().reshape(side_length, side_length,3)
        plt.imshow(y)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # test_motion_xyt()
    test_motino_phase(0.0)