import copy
from typing import List
import numpy as np
import torch
from models import *
from torch import nn
import torchvision.models as models
from py_scripts import LightningDataModule
from .model_params import *
from .dataset_params import *
import ipdb
import pytorch_lightning as pl

def reload_params_from_saved_model(full_cpkt, params, experiment_param_modifications, dataset_params ):

    # overwrite with the new dataset and specific experiment parameters that are defined. 

    # delete old model foundation
    dc_full_cpkt_hparams = copy.deepcopy(full_cpkt["hyper_parameters"])
    del dc_full_cpkt_hparams['model_class']
    
    # at some point concluded this was necessary? 
    #dc_params = copy.deepcopy(params)
    #dc_dataset_params = copy.deepcopy(dataset_params) 
    #dc_experiment_param_modifications = copy.deepcopy(experiment_param_modifications)

    # this will ensure that the custom params dominate
    dc_full_cpkt_hparams.update(dataset_params)
    dc_full_cpkt_hparams.update(experiment_param_modifications)
    
    # update parameters with model loaded in ones. 
    params.update(dc_full_cpkt_hparams)

    return params

def set_model_output_size(params, dataset_params):
    if params["classification"]:
        params["output_size"] = dataset_params["nclasses"]
    else: 
        params["output_size"] = dataset_params["input_size"]

def get_params_net_dataloader(
    model_name,
    dataset_name,
    load_from_checkpoint=None,
    dataset_path="data/",
    verbose=True,
    experiment_param_modifications=None,
    ):


    """
    Returns model parameters for given model_style and an optional list of regularizers. 
    """
    #if load_from_checkpoint is None:
    params = copy.deepcopy(global_default_settings)
    params.dataset_name = dataset_name
    params.model_name = model_name

    if "load_sdm_default_params" in model_params[model_name] and model_params[model_name].load_sdm_default_params:
        params.update(default_sdm_model_settings)
    
    params.update( model_params[model_name] )
    params.update( dataset_params[dataset_name] )
    d_params = dataset_params[dataset_name]

    # set up here so it applies automatically and is not over written by any loaded in model (custom kwards overwrites the loaded in model parameters.)

    #TODO: this is kind of hacky but helps ensure no errors during the run because I forget to set the wrong master parameters.
    if "CachedOutputs" in params['torchified_dataset_suffix']:
        params.log_receptive_fields = False
        params.log_model_predictions = False

    # THIS NEEDS TO BE THE LAST THING THAT IS MODIFIED
    if verbose: 
        print("Custom args are:", experiment_param_modifications)
    for key, value in experiment_param_modifications.items():
        params[key] = value

    # used for the custom faster torch datasets
    dataset_path+= params.torchified_dataset_suffix

    params.act_string = str(params.act_func)
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.load_from_checkpoint = load_from_checkpoint

    ##########
    # loading in model
    if load_from_checkpoint:  
        import os
        print("Load from checkpoint", load_from_checkpoint, os.getcwd())
        full_cpkt = torch.load(load_from_checkpoint, map_location=params.device)


        if not params.load_just_state_dict:
        
            # use loaded model settings aside from those from the experiment and the dataset. 
            # later temporarily reset specific model params that may have now changed to load in the model and then post hoc edit it. 
            params = reload_params_from_saved_model(full_cpkt, params, experiment_param_modifications, d_params )

    set_model_output_size(params, d_params)

    # ensure the device is not overwritten no matter what. 
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting random seed. Want to do this before the models are initialized. 
    if params.random_seed is not None: 
        pl.utilities.seed.seed_everything(seed=params.random_seed, workers=False)


    # getting whatever was set from either the loaded in model or the dataset: 
    
    # needs to be able to handle the model loading. 
    if params["classification"] and "accuracy" not in params.metrics_to_log:
        params.metrics_to_log.append('accuracy')
    elif not params["classification"] and "accuracy" in params.metrics_to_log:
        params.metrics_to_log.remove('accuracy')


    # converting number of update steps into a k_transition and gaba activation coutner

    if "k_approach" in params and "GABA_SWITCH" in params['k_approach'] and "num_binary_activations_for_gaba_switch" in params.keys() and params["num_binary_activations_for_gaba_switch"] is None:

        params["num_binary_activations_for_gaba_switch"] = params['num_pre_gaba_switch_neuron_update_steps']*params['batch_size']*params['sdm_neuron_multiheads']
        print("SDM Neuron multiheads is:", params['sdm_neuron_multiheads'])
        
        if "num_data_points" in params:
            print("GABA switch will happen by approx. epoch:", params['num_pre_gaba_switch_neuron_update_steps']/(params['num_data_points']/(params['batch_size'] )) )

    if "k_transition_epochs" in params.keys() and params["k_transition_epochs"] is None:
        params["k_transition_epochs"] = int(params["epochs_to_train_for"] / 2)
    
    data_module = LightningDataModule(
        params,
        data_path=dataset_path
    )

    if load_from_checkpoint: 
        #params_pre_load_model = copy.deepcopy(params)

        # need to preserve loaded in model output size this uses the original 

        if params.load_just_state_dict:
            params['output_size'] = params.original_model_output_size

        else: 
            for og_model_setting in ['output_size', 'use_bias_output']:
                params[og_model_setting] =full_cpkt["hyper_parameters"][og_model_setting]

    model = params.model_class(params)

    if load_from_checkpoint is not None:
        if verbose: 
            print("!!! Loading in model with checkpoint saved parameters!!!")

        if not params.load_just_state_dict and "X_a.weight" in full_cpkt['state_dict']:
            # so I can load in older models.  
            for remove_vals in ['X_a.weight', 'X_a.bias', 'X_vT.weight', 'X_vT.bias']:
                del full_cpkt['state_dict'][remove_vals]


        if params.load_just_state_dict:  
            model.load_state_dict(full_cpkt) 
            params.load_existing_optimizer_state = False
        else: 
            model.load_state_dict( full_cpkt['state_dict'])

        # either it already agrees or it will now. 
        set_model_output_size(params, d_params)

        if params.model_name == "CONVMIXER" and model.classifier.weight.shape[0]!=params.output_size:
            # Reset the output head: 
            # need to update the 
            model.classifier = nn.Linear(params.hdim, params.output_size, bias=params.use_bias_output)

        if "epoch" in full_cpkt:
            params.starting_epoch=full_cpkt["epoch"]
            if params.load_existing_optimizer_state:
                params.epochs_to_train_for += params.starting_epoch

        del full_cpkt

    if verbose: 
        print(
            "Number of unique parameters trained in the model",
            len(list(model.parameters())),
        )

    non_zero_weights = 0
    for p in list(model.parameters()):
        if len(p.shape)>1: 
            non_zero_weights+= (torch.abs(p)>0.0000001).sum()
    if verbose: 
        print("Number of non zero weights is:", non_zero_weights)
        print("Final params being used", params)

    if params.log_image_data_every_n_epochs>1:
        assert params.log_image_data_every_n_epochs%params.check_val_every_n_epoch==0, "need for overall validation logging interval to be a fraction of the receptive field logger. Else will never log!"

    # useful for wandb later. 
    params.first_layer_nneurons = params.nneurons[0]

    return params, model, data_module