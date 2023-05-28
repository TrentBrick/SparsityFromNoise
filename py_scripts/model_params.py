import copy
from enum import Enum, auto
from typing import List
import numpy as np
import torch
from models import *
from torch import nn
from py_scripts import LightningDataModule
from easydict import EasyDict

global_default_settings = EasyDict(

    # task settings
    classification = True,
    batch_size=512,
    epochs_to_train_for=200,

    # training config
    num_workers=0,
    early_stopping=False,

    separate_bias_opts = False, 
    separate_bias_opts_lr = 0.5,
    
    # optimizer/epoch
    starting_epoch = 0, 
    load_existing_optimizer_state = True,
    opt="Adam",  # Stochastic Grad Descent with Momentum. # Adam
    lr=0.0001,
    sgdm_momentum=0.9,
    adam_betas = (0.9, 0.999),
    gradient_clip=1.0,
    adamw_l2_loss_weight=0.001,

    lr_scheduler = None,  # 'StepLR'
    lr_scheduler_step_size = 30,
    step_lr_gamma = 0.1,

    # Network settings
    use_sdm = False, 
    nneurons=[1000],
    act_func=nn.ReLU(),
    use_bias_hidden = True, 
    use_bias_output=True,
    use_shared_bias = False,
    noise_type="normal",
    noise_layers_throughout_model=True,

    # Dataset
    dataset_size = None,
    normalize_n_transform_inputs =False, 
    use_convmixer_transforms=False,
    min_max_scaler = False, 
    noise_off_during_eval = False, 

    # Logging
    metrics_to_log = ["loss", "accuracy"],
    check_val_every_n_epoch = 10, # how often to run the validation loop
    save_model_checkpoints = True, 
    num_checkpoints_to_keep = 1, # for saving the model
    checkpoint_every_n_epochs = 10,
    use_wandb = True, 
    log_model_predictions = True, 
    log_receptive_fields = False,
    log_image_data_every_n_epochs = 10, # receptive fields and or model predictions

    num_receptive_field_imgs=10,
    num_cnn_receptive_field_imgs=10,
    num_task_attempt_imgs = 10, 

    random_seed = None, 

    load_just_state_dict = False, 

    cifar10_mean = (0.4914, 0.4822, 0.4465),
    cifar10_std = (0.2471, 0.2435, 0.2616),

    use_inhib_circuit = False, #"ALL-TO-ALL", #"INTERNEURON"
    only_inhibitory = False, # will apply ReLU activation and do a subtraction operation
    num_inhib_circuit_steps = 1,
    inhib_circuit_step_size = 1.0,

    # Regularization terms. All off by default
    l2_loss_weight=None,  # This is set lower down by the regularizer region
    l1_loss_weight=None,
    dropout_prob=0.0,

    activation_l1_coefficient = None,
    non_relu_act_threshold = 0.0001, 
)

default_sdm_model_settings = EasyDict(
    use_sdm=True,
    k_approach = "LINEAR_DECAY_MASK", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH_ACT_BIN", "LEARN_K_SIGMOID", "LEARN_K_REINFORCE", "LEARN_K_SHARED_BIAS"

    learn_k_init = 10, # for ehen it is being learnt
    learn_k_window_perc = 0.1, 
    learn_k_lr_multiplier = 10, 

    num_binary_activations_for_gaba_switch = None , # will be calculated as a function of dataset size. 
    num_pre_gaba_switch_neuron_update_steps = 1000,
    k_transition_epochs = None,
    sdm_neuron_multiheads = 1, # number of times neuorns see data (eg convolution operations)

    use_bias_hidden = False, 
    use_bias_output=False,
    can_enforce_positive_weights_on_bias_terms = True, 
    norm_addresses=True,
    norm_values=False,
    all_positive_weights = False, 
    learn_addresses= True,
    top_k_blocks = 1,
    
)

# these will all overwrite the defaults. 
model_params = EasyDict(

    DIFFUSION_SDM = dict(

        use_sdm=True, 

        model_class = SDM_DIFFUSION,

        load_sdm_default_params = True,

        opt='SparseAdam',
        lr=0.0001,
        k_transition_epochs = 100,

        batch_size=512, 
        #dataset_size = 10,

        nneurons=[1000],

        use_projection_matrix = False, 
        project_before_noise = False, 

        adjust_diffusion = False,
        epoch_to_start_noise_annealing = None,
        epochs_to_terminal_noise = None,
        terminal_diffusion_noise = 0.0,

        k_min=10, 
        k_approach = "LINEAR_DECAY_MASK", #"FLAT_MASK", "FLAT_SUBTRACT", "LINEAR_DECAY_MASK", "LINEAR_DECAY_SUBTRACT", "GABA_SWITCH_ACT_BIN", "LEARN_K_SIGMOID", "LEARN_K_REINFORCE","LEARN_K_SHARED_BIAS"
        
        all_positive_weights = False,

        use_bias_hidden = False,
        use_bias_output = False, 
        transpose=False, 

        norm_addresses=True, 
        norm_values = False, 
        norm_activations=False,

        #preprocessing_type = "SIMPLE",
        log_receptive_fields = False,
        
        classification=False, 

    ), 

    DIFFUSION_FFN = dict(

        model_class = FFN_DIFFUSION,

        opt='Adam',
        lr=0.0001,

        batch_size=512, 
        #dataset_size = 10,

        nneurons=[1000],
        use_projection_matrix = False, 
        project_before_noise = False, 
        adjust_diffusion = False, 

        nconvergence_steps=1,
        diffusion_noise=0.1,

        use_bias_hidden = True,
        use_bias_output = True,
        use_shared_bias = False,  
        transpose=False, 

        #preprocessing_type = "SIMPLE",
        log_receptive_fields = False,
        log_model_predictions=False,
        classification=False, 
    ), 

    DIFFUSION_ATTN = dict(

        model_class  = ATTN_DIFFUSION, 

        opt='SparseAdam',
        lr=0.01,

        batch_size=512, 
        #dataset_size = 10,
        diffusion_noise=0.1,

        auto_assoc = True, 
        l2_norm_k_and_q = True, 

        beta_init = 2,
        
        log_receptive_fields = False,
        log_model_predictions=False,
        classification=False, 

    ), 

    RESNET50 = dict(
        model_class  = ResNet50, 
        batch_size=1024,
        log_receptive_fields = False,
        load_just_state_dict = True,

        wide_resnet=False, 

        check_val_every_n_epoch = 5,

        normalize_n_transform_inputs = True, 
        use_convmixer_transforms = False, 

    ),

    CONVMIXER =dict(
        # not implementing learning rate schedule for now. 
        # or the gradient scaler. 
        model_class  = ConvMixer, 
        load_just_state_dict = True, 
        original_model_output_size = 1000,

        batch_size = 512,
        #lr_schedule_triangular =True, 
        kernel_size=5, 
        patch_size=2,
        gradient_clip=1.0,
        hdim=256,
        nblocks=8,
        adamw_l2_loss_weight=0.005,
        lr=0.01, 
        opt="AdamW",
        
        normalize_n_transform_inputs = False, 
        use_convmixer_transforms = True,  
        
        log_receptive_fields=False,
    ),

)