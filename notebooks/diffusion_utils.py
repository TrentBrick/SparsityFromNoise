

import glob
import torch 
import numpy as np
from easydict import EasyDict
import matplotlib.pyplot as plt
from py_scripts import LightningDataModule, get_params_net_dataloader

import torchvision

def fit_beta_and_bias(x, y):
    beta = np.cov(x, y)[0][1] / np.var(x)
    bias = np.mean(y) - (beta*np.mean(x))
    return beta, bias

def cosine_sim_matrices(a,b, dim=1):
    a = a/torch.norm(a, dim=dim, keepdim=True)
    b = b/torch.norm(b, dim=dim, keepdim=True)
    return a @ b.T


def cosine_sim(a,b, dim=1):
    a = a/torch.norm(a, dim=dim, keepdim=True)
    b = b/torch.norm(b, dim=dim, keepdim=True)
    return torch.sum(a * b, dim=dim)

def euc_dist(a,b,dim=1):
    return torch.sum( (a-b)**2, dim=dim).sqrt()

def get_active_neurons(model, all_data, device, nneurons):

    batch_size = 1000 if torch.cuda.is_available() else 100

    activation_summer = torch.zeros( nneurons ).to(device)
    
    with torch.no_grad():
        for i in range( len(all_data)//batch_size ):
            if i%5==0:
                print("Batch", i)

            sind = i*batch_size
            eind = sind+batch_size

            x = all_data[sind:eind].to(device)

            active_neurons = model.run_model_till_neuron_activations(x)

            activation_summer += active_neurons.sum(0)

    return activation_summer


def min_max_scale(x):
    min_vals, _ = x.min(dim=1)
    max_vals, _ = x.max(dim=1)
    min_vals = min_vals.unsqueeze(1)
    max_vals=max_vals.unsqueeze(1)

    return (x+min_vals.abs()) * 1/(max_vals-min_vals)

def peak_scale(x):
    #x = min_max_scale(x)
    peak_vals, _ = x.abs().max(dim=1)
    scaler = (0.5/peak_vals).unsqueeze(1)
    return (scaler*x) + 0.5


def gridshow(imgs, title=None, nrow=5, imgsize=(20,20), nimages=None, use_mm_scale=False, use_peak_scale=False, reshape=False, side_dim=32, save_name=None, show_plot=True):

    assert int(use_mm_scale)+int(use_peak_scale)<2, "Cant have both on!"

    if use_mm_scale:
        imgs = min_max_scale(imgs)
        reshape = True # as need to minmax it first while it is flat.
    if use_peak_scale:
        imgs = peak_scale(imgs)
        reshape = True

    if reshape:
        imgs = imgs.view(len(imgs), 3,side_dim,side_dim)
    if nimages is None: 
        nimages = len(imgs)
    fig = plt.gcf()
    fig.set_size_inches(imgsize)
    grid_img = torchvision.utils.make_grid(imgs[:nimages].cpu(), nrow=nrow)
    plt.imshow(grid_img.permute(1, 2, 0))
    if title is not None: 
        plt.title(title)

    if save_name: 
        plt.gcf().savefig(f'figs/{save_name}.png', dpi=250)

    if show_plot:
        plt.show()

def imshow(img, title="No Title Given", save_name=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    if save_name: 
        plt.gcf().savefig(f'figs/{save_name}.png', dpi=250)
        
    plt.show()

def load_model(vae_run_name, dataset_path, save_dir, device, extra_extras=False, specific_epoch=""):
    extras = dict(
        epochs_to_train_for=10,
        num_workers=1, 
        device=device,
        dataset_path="../data/", 
        log_topk_metrics=False  ,
        check_val_every_n_epoch=1,
        min_max_scaler = False,
    )

    if extra_extras:
        extras.update(extra_extras)

    src = f"../../scratch_link/Foundational-SDM/wandb_Logger/{vae_run_name}/version_None/checkpoints/*{specific_epoch}*"
    print("looking for:", src)
    model_checkpoints = glob.glob(src)
    print("Checkpoints found", model_checkpoints)
    model_path = model_checkpoints[-1]
    #print("Using model path", model_path)
    ckpt = torch.load(model_path, map_location=device)
    params = ckpt['hyper_parameters']
    params = EasyDict(params)

    _, model, _ = get_params_net_dataloader(params.model_name, params.dataset_name,    load_from_checkpoint=model_path, verbose=False, experiment_param_modifications=extras)
    model.eval()

    model = model.to(device)
    model.eval()

    return model, params 

def get_model_latents(model, params, device):

    dataset_path="../data/"
    params.min_max_scaler = False
    data_module = LightningDataModule(
                params,
                data_path=dataset_path+params.torchified_dataset_suffix
            )
    data_module.setup(None, train_shuffle=False, test_shuffle=False)

    all_data = data_module.train_data.images
    all_labels = data_module.train_data.img_labels

    if all_data.dtype is torch.uint8:#"/ImageNet32/" in self.dataset_path or "/CIFAR10/" in self.dataset_path:
        all_data = all_data.type(torch.float)/255

    batch_size = 1000 if torch.cuda.is_available() else 100
    out_cats = dict()
    with torch.no_grad():
        for i in range( len(all_data)//batch_size ):
            if i%5==0:
                print("Batch", i)
            
            sind = i*batch_size
            eind = sind+batch_size
            
            x = all_data[sind:eind].to(device)
            
            if params.model_style.name == "RITHESH_VQ_VAE" or params.model_style.name == "MISHA_VQ_VAE":
                outs = model.forward(x, output_indices=True )
                
            elif params.model_style.name == "VAE_RESID_CNN":
                outs = model.forward(x)
            else: 
                outs = model.forward(x, output_model_data=True, return_cosine_thresholds=True)
                # x, mu, log_sigma_approx, model_data_dict, cosine_thresholds
            
            for vind, v in enumerate(outs):
                
                if type(v) is dict:
                    v = v['post_acts'].cpu()
                else: 
                    v = v.cpu()
                
                if i==0:
                    out_cats[vind] = v
                else: 
                    out_cats[vind] = torch.cat( [out_cats[vind],v],dim=0)

    return out_cats 