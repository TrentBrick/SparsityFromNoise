# simple test script to make sure that everything is workign or easy debugging: 

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb
from py_scripts.combine_params import *
import random 
import numpy as np 

model_name = "DIFFUSION_FFN" #"DIFFUSION_FFN"#"DIFFUSION_ATTN"
dataset_name = "Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10"

load_path = None

output_directory = "FILL-IN"

extras = dict(

    epochs_to_train_for = 500, 
    num_workers=0, 
    random_seed = 5,
    nneurons=[1000],
    opt='Adam',
    diffusion_noise=0.8,
    use_wandb=False, 

)

if load_path:
    print("LOADING IN A MODEL!!!")

model_params, model, data_module = get_params_net_dataloader(model_name, dataset_name, load_from_checkpoint=load_path, experiment_param_modifications=extras)

if model_params.use_wandb:
    wandb_logger = WandbLogger(project="Diffusion-SDM", entity="", save_dir=output_directory)
    model_params.logger = wandb_logger
else: 
    model_params.logger = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("using cuda", device)
    gpu = [0]
else: 
    print("using cpu", device)
    gpu=None

# SETUP TRAINER
if model_params.load_from_checkpoint and model_params.load_existing_optimizer_state:
    fit_load_state = load_path
else: 
    fit_load_state = None

callbacks = []
checkpoint_callback = False

temp_trainer = pl.Trainer(
        #precision=16, 
        logger=model_params.logger,
        max_epochs=model_params.epochs_to_train_for,
        check_val_every_n_epoch=1,
        num_sanity_val_steps = False,
        enable_progress_bar = True,
        gpus=gpu, 
        callbacks = callbacks,
        enable_checkpointing=False,
        #checkpoint_callback=checkpoint_callback, # dont save these test models. 
        #limit_train_batches=10,
        #profiler="simple" # if on then need to set epochs_to_train_for to a v low score.
        )
temp_trainer.fit(model, data_module)#, ckpt_path=False)#fit_load_state)
wandb.finish()
