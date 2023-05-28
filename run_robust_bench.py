# !pip install git+https://github.com/RobustBench/robustbench@v0.2.1
#from robustbench.utils import load_model
# Load a model from the model zoo
#model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra',dataset='cifar10',threat_model='Linf')
import torch
import torch.nn as nn
from py_scripts.combine_params import *
from torchvision import transforms

class Preprocessing_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.cifar10_mean = (0.4914, 0.4822, 0.4465)
        self.cifar10_std = (0.2471, 0.2435, 0.2616)

        self.norm = transforms.Normalize(self.cifar10_mean, self.cifar10_std) 

    def forward(self, x):
        x = self.norm(x)
        return self.model(x)

device = torch.device("cuda:0")

model_name = "CONVMIXER"
dataset_name = "CIFAR10"

prefix = "../scratch_link/Foundational-SDM/wandb_Logger/"
train_name = "3.0ConvMixer_FixedAug_NoPretrain_YesAugs_Adv_Robustness_ClassifCIFAR10_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"
blah = "version_None/checkpoints"
epoch_and_step = "epoch=19-step=7820.ckpt"
load_path = f"{prefix}{train_name}/{blah}/{epoch_and_step}"
extras = dict(
    load_just_state_dict = False, 
    diffusion_noise = 0.0
)

model_params, model, data_module = get_params_net_dataloader(model_name, dataset_name, load_from_checkpoint=load_path, experiment_param_modifications=extras)

model.eval()

# just for good measure. 
model.noise_layer.turn_off_diffusion_noise()

print("Noise in the model is:::", model.noise_layer.noise_amount)

#model = Preprocessing_Wrapper(model)

# Evaluate the Linf robustness of the model using AutoAttack
from robustbench.eval import benchmark
clean_acc, robust_acc = benchmark(model,
                                  dataset='cifar10',
                                  threat_model='Linf',
                                  eps=8/255,
                                  batch_size = 128,
                                  device=device
)

print("clean and robust accuracies",clean_acc, robust_acc )