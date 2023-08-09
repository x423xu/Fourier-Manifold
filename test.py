import torch
import cv2
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
# from motion_phase import PLFourierNet
import numpy as np

def test_motion_xyt():
    from motion_phase import PLFourierNet
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
    from motion_phase import PLFourierNet
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

def test_motion_sep():
    from motion_septraining import PLFourierNet, HDRDataModule
    import pytorch_lightning as pl
    kwargs = {
        'project_name': 'motion_sep_training',
        'batch_size': 1,
        'side_length': 256,
        'pos_embed': False,
        'sequence_range': [15,175],
        'skip': 10,
        'motion_path': './motion_data/man'
    }
    process_id = 2
    ckpt_path = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p{}.ckpt".format(process_id)
    model = PLFourierNet()
    model = model.load_from_checkpoint(ckpt_path)
    data_module = HDRDataModule(batch_size = kwargs['batch_size'], side_length=kwargs['side_length'], motion_path = kwargs['motion_path'], sequence_range=kwargs['sequence_range'], skip = kwargs['skip'], process_id=process_id)
    trainer = pl.Trainer( max_epochs=300, devices = 1, log_every_n_steps=1, enable_progress_bar=False)
    trainer.test(model, data_module)

def plot_weights():
    import torch
    num = 16
    state_dicts  = []
    for i in range(num):
        ckpt_path = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p{}.ckpt".format(i)
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dicts.append(state_dict)
    for k in state_dicts[0].keys():
        print(k)
        if 'weight' in k:
            plt.figure()
            for i in range(num):
                weights = state_dicts[i][k]        
                plt.subplot(4,4,i+1)
                bins,_,_ = plt.hist(weights.detach().cpu().numpy().flatten(), bins=100)
                plt.xlim(weights.min(), weights.max())
                plt.ylim(0, max(bins))
                plt.title('p{}'.format(i))
            plt.savefig('output/hist_{}.png'.format(k))

            plt.figure()
            for i in range(num):
                weights = state_dicts[i][k].detach().cpu().numpy().flatten()       
                plt.subplot(4,4,i+1)
                plt.plot(np.arange(weights.shape[0]), weights)
                # plt.scatter(np.arange(weights.shape[0]), weights)
                # plt.bar(np.arange(weights.shape[0]), weights)
                # plt.xlim(weights.min(), weights.max())
                # plt.ylim(0, max(bins))
                plt.title('p{}'.format(i))
            plt.savefig('output/plot_{}.png'.format(k))
            plt.close('all')
def test_interpolation():
    from motion_septraining import PLFourierNet
    import torch

    ckpt_path1 = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p0.ckpt"
    ckpt_path2 = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p10.ckpt"
    model1 = PLFourierNet()
    model1 = model1.load_from_checkpoint(ckpt_path1)

    model2 = PLFourierNet() 
    model2 = model2.load_from_checkpoint(ckpt_path2)

    #interpolate state dict
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    for k in state_dict1.keys():
        if 'linear' in k and 'filters' not in k:
            state_dict1[k] = (state_dict1[k] + state_dict2[k])/2
        # elif 'filters' in k:
        #     state_dict1[k] = state_dict2[k]
        # if 'linear' in k and 'filters' in k and 'weight' in k:
        #     state_dict1[k] = state_dict2[k]
    model1.load_state_dict(state_dict1)
    side_length = 256
    x = torch.stack(
        torch.meshgrid(
            [
                torch.linspace(-1.0, 1.0, side_length),
                torch.linspace(-1.0, 1.0, side_length),
            ]
        ),
        dim=-1,
    ).view(1,-1,2).float()
    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)

    y1 = (y1-y1.min())/(y1.max()-y1.min()).squeeze()
    y2 = (y2-y2.min())/(y2.max()-y2.min()).squeeze()
    y1 = y1.detach().cpu().numpy().reshape(side_length, side_length,3)
    y2 = y2.detach().cpu().numpy().reshape(side_length, side_length,3)
    plt.subplot(1,2,1)
    plt.imshow(y1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(y2)
    plt.axis('off')
    plt.show()

def get_eigen():
    import numpy as np
    import os
    num = 16
    if os.path.exists('state_dicts.npy'):
        state_dicts = np.load('state_dicts.npy')
    else:
        state_dicts  = []
        for i in range(num):
            sti = []
            ckpt_path = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p{}.ckpt".format(i)
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            for k, v in state_dict.items():
                if 'weight' in k and 'linear' in k and 'filters' not in k and 'output' not in k:
                    sti.append(v.detach().cpu().numpy())
            state_dicts.append(sti)
        state_dicts = np.array(state_dicts)
        if not os.path.exists('state_dicts.npy'):
            np.save('state_dicts.npy', state_dicts)

    import torch 
    import torch.nn as nn
    from tqdm import tqdm
    torch.manual_seed(20)
    state_dicts = torch.from_numpy(state_dicts).cuda()
    print(state_dicts.shape)
    st1 = state_dicts[0].view(1,-1)
    st2 = state_dicts[10].view(1,-1)
    A = nn.Linear(st1.shape[1], 1, bias=False)
    A.cuda()
    A.requires_grad_(True)
    E = torch.zeros(1,).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam(list(A.parameters()) + [E], lr=1e-3)
    pbar = tqdm(range(10000))
    loss_min = 1000
    for i in pbar:
        loss = (A(st1)-E*A(st2)).abs().mean()
        if loss.item() < loss_min:
            loss_min = loss.item()
            torch.save(A.state_dict(), 'A.pth')
            torch.save(E, 'E.pth')
            e_min = E.item()
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            pbar.set_description('step {}, loss {}, Eigen value {}'.format(i, loss_min, e_min))

def test_eigen():
    import torch
    A = torch.load('A.pth')
    E = torch.load('E.pth')

    from motion_septraining import PLFourierNet
    import torch

    ckpt_path1 = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p0.ckpt"
    ckpt_path2 = "/home/xxy/Documents/code/Fourier-Manifold/motion_sep_training/checkpoints/20230807_18:19:58p10.ckpt"
    model1 = PLFourierNet()
    model1 = model1.load_from_checkpoint(ckpt_path1)

    model2 = PLFourierNet() 
    model2 = model2.load_from_checkpoint(ckpt_path2)

    #interpolate state dict
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    for k in state_dict1.keys():
        if 'linear' in k and 'filters' not in k:
            state_dict1[k] = (state_dict1[k] + state_dict2[k])/2
        # elif 'filters' in k:
        #     state_dict1[k] = state_dict2[k]
        # if 'linear' in k and 'filters' in k and 'weight' in k:
        #     state_dict1[k] = state_dict2[k]
    model1.load_state_dict(state_dict1)
    side_length = 256
    x = torch.stack(
        torch.meshgrid(
            [
                torch.linspace(-1.0, 1.0, side_length),
                torch.linspace(-1.0, 1.0, side_length),
            ]
        ),
        dim=-1,
    ).view(1,-1,2).float()
    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)

    y1 = (y1-y1.min())/(y1.max()-y1.min()).squeeze()
    y2 = (y2-y2.min())/(y2.max()-y2.min()).squeeze()
    y1 = y1.detach().cpu().numpy().reshape(side_length, side_length,3)
    y2 = y2.detach().cpu().numpy().reshape(side_length, side_length,3)
    plt.subplot(1,2,1)
    plt.imshow(y1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(y2)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # test_motion_xyt()
    # test_motino_phase(0.0)
    # test_motion_sep()
    # plot_weights()
    # test_interpolation()
    get_eigen()