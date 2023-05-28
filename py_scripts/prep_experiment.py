# shared by ray and slurm runners
import copy 
import wandb 
from py_scripts.combine_params import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import numpy as np 
import random
import torch

output_directory = "NEED-T0-FILL-IN"

wandb_project = "Diffusion-SDM"

def init_exp_settings(exp_ind,job_script):
    # takes in a job script
    exp_settings = job_script.exp_list[exp_ind]
    settings_for_all = copy.deepcopy(job_script.settings_for_all)
    settings_for_all.update(exp_settings)
    exp_settings = settings_for_all

    if "test_name" in exp_settings.keys():
        exp_settings['name_prefix'] = copy.deepcopy(exp_settings['test_name'])
    if job_script.name_suffix:
        exp_settings['name_suffix'] = job_script.name_suffix
        exp_settings['test_name']+=job_script.name_suffix

    return exp_settings

def compile_experiment(exp_settings, num_workers):

    model_name = exp_settings["model_name"]
    exp_settings.pop("model_name")

    if "dataset_name" in exp_settings.keys():
        dataset_name = exp_settings["dataset_name"]
        exp_settings.pop("dataset_name")

    print("Init experiment", model_name, "special params:", exp_settings)

    if "load_path" in exp_settings.keys():
        print("LOADING IN A MODEL!!!")
    else:
        exp_settings["load_path"] = None

    exp_settings["num_workers"] = num_workers

    params, model, data_module = get_params_net_dataloader(
        model_name, dataset_name, load_from_checkpoint=exp_settings["load_path"], experiment_param_modifications=exp_settings
    )

    tags = [model_name] #params.opt, # dataset_name.name, 
    if "k_approach" in exp_settings.keys():
        tags.append(exp_settings['k_approach'])

    if "test_description" not in exp_settings.keys():
        exp_settings["test_description"] = None

    if "test_name" not in exp_settings.keys():
        exp_settings["test_name"] = None

    if params.use_wandb:
        wandb_logger = WandbLogger(
            project="Diffusion-SDM",
            entity="",
            save_dir=output_directory,
            tags=tags,
            notes=exp_settings["test_description"],
            name=exp_settings["test_name"],
        )
    else: 
        wandb_logger = None

    params.logger = wandb_logger

    # GET MODEL
    if params.load_from_checkpoint and params.load_existing_optimizer_state:
        params.fit_load_state = exp_settings["load_path"]
    else: 
        params.fit_load_state = None

    callbacks = []
    if params.early_stopping:
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0, patience=5, verbose=False, mode="min"
        )
        callbacks.append(early_stop_callback)

    if params.save_model_checkpoints:
        model_checkpoint_obj = pl.callbacks.ModelCheckpoint(
            every_n_epochs = params.checkpoint_every_n_epochs,
            save_top_k = params.num_checkpoints_to_keep
        )
        callbacks.append(model_checkpoint_obj)
        checkpoint_callback = True
    else: 
        checkpoint_callback = False


    '''# setting random seed
    if params.random_seed is not None: 
        print("Setting random seed to be:", params.random_seed)
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)'''

    return model, data_module, params, callbacks, checkpoint_callback


