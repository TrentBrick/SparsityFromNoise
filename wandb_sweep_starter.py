import wandb 
import os
import sys 
from py_scripts.prep_experiment import wandb_project

config_dict={
    "name": "TopKApproaches",
    "metric": {
        "name": "val/accuracy", 
        "goal": "minimize"
    },
    "method": "random", # 'random', 'grid', or 'bayes'
}

# careful with value vs values!

sweep_params = {
    'epochs_to_train_for':{
        "value": 800 # if less than 500 results may not be as meaningful
    },
    "model_name": {
        "value": "DIFFUSION_SDM"
    },
    "dataset_name": {
        "value": "Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10"
    },
    "diffusion_noise": {
        "value": 0.8
    },
    "nneurons": {
        "value": [10000]
    },
    "k_min": {
        "values": [10, 50, 100, 300, 500, 1000, 3000]
    },
    "k_approach": {
        "values": [
            "FLAT_SUBTRACT", 
            "FLAT_MASK", 
            "LINEAR_DECAY_MASK",
            "LEARN_K_SIGMOID",
            "LEARN_K_REINFORCE",
        ]
    },
    "learn_k_init": {
        "value": 100
    },
    "k_transition_epochs": {
        "value": 400
    },
    "num_pre_gaba_switch_neuron_update_steps": {
        "value": 10_000
    },
    "use_bias_hidden": {
        "value": True
    },
    "use_bias_output": {
        "value": True
    },
    "norm_addresses": {
        "value": False
    },
    "norm_values": {
        "value": False
    },
    "norm_activations": {
        "value": False
    },
    
}
if __name__ == '__main__':
    config_dict["parameters"] = sweep_params
    sweep_id = wandb.sweep(config_dict, project=wandb_project, entity="")
    os.system(f"cp wandb_sweep_starter.py scripts_hparam_search/{config_dict['name']}-sweep_starter.py")
    print(sweep_id)
    sys.exit(0)
