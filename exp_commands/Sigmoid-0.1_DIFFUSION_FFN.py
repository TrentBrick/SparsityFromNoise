import os
import sys
import copy
import torch 

settings_for_all = dict(
    epochs_to_train_for = 6000,
    model_name= "DIFFUSION_FFN",
    dataset_name = "Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10", #CIFAR10,#"Cached_ConvMixer_WTransforms_ImageNet32_CIFAR10", #!!

    opt='Adam',#SparseAdam',
    lr=0.0001,#0.0001,#0.15,
    
    batch_size=1024,#1000, 
    dataset_size = None,#1000,

    nneurons=[10000],

    diffusion_noise=0.0,
    adjust_diffusion=False,

    use_projection_matrix=False,
    project_before_noise = False,
   
    use_shared_bias = False, 

    transpose=False, 
    classification=False, 

    log_receptive_fields = False,
    log_model_predictions=False, 
    
)

main_title = "Sigmoid-0.1"

epoch_and_step = "epoch=5989-step=293510.ckpt"#"epoch=799-step=39200.ckpt"
exp_name_path = "ReconCIFAR10Long_EvenMore_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"#"Poisson_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1" #"Baseline_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"

load_all_models = False

modifications = dict(

    act_func = torch.nn.Sigmoid(),
    non_relu_act_threshold = 0.01,
    #dataset_name = "CIFAR10",

    #model_name= "DIFFUSION_SDM",
    #norm_addresses=True, 
    #norm_values = False, 
    #norm_activations=False,
    #k_approach = "FLAT_SUBTRACT", #"LEARN_K_SHARED_BIAS"
    #k_min = 10,
    #use_bias_hidden = True,
    #use_bias_output = True, 
    #norm_addresses=False, 
    #norm_values = False, 
    #norm_activations=False,

    #continual_learning = True, 
    
    #model_name= "CONVMIXER",
    #classification = True, 
    #batch_size=128,

    #opt='SGD',#SparseAdam',
    #lr=0.5,#0.0001,#0.15,
    #separate_bias_opts = True,
    #separate_bias_opts_lr = 0.1,
    #nneurons=[1_000],

    #random_seed = 5,

    #use_inhib_circuit = "ALL-TO-ALL", #"ALL-TO-ALL", #"INTERNEURON"
    #only_inhibitory = False, # will apply ReLU activation and do a subtraction operation
    #num_inhib_circuit_steps = 1,
    #inhib_circuit_step_size = 1.0,

    #adjust_diffusion = True,
    #epochs_to_train_for = 2000,
    #epoch_to_start_noise_annealing = 800,
    #epochs_to_terminal_noise = 800,
    #terminal_diffusion_noise = 0.0,

    #use_auto_attack = True, 

    #load_path = "saved_models/ConvMixer_ImgNet32_WTransforms_Just_State_Dict",

    #normalize_n_transform_inputs = False, 
    #use_convmixer_transforms = True,

    #adversarial_eval_only = True, 
    #pgd_step_iters=30,
    #adversarial_max_distortion = 0.2,#0.03,

    #split_random_seed = None, #3,15,27,97
    #use_bias_output = False, 
    #epochs_per_cl_task = 50,  
    #epochs_to_train_for = 250,

    
    #opt='SparseAdam',
    #lr=0.00001,

   
    #noise_type = "poisson"

    #use_explain_away = True, 
    #explain_away_lr = 0.005,
    #explain_away_opt_z_n_grad_steps = 30,
    
    #adversarial_train = True, 
    #epoch_to_start_adversarial_training = 0,

    #adjust_diffusion=True,

    #model_name= "DIFFUSION_SDM",
    #norm_addresses=True, 
    #norm_values = False, 
    #norm_activations=False,
    #k_approach = "LEARN_K_SHARED_BIAS"

    #"LEARN_K_SIGMOID",
    #learn_k_init=1000,
    #learn_k_window_perc = 0.05, 
    #learn_k_lr_multiplier = 200,  
    
    #classification=True, 
    #use_projection_matrix=True,
    #project_before_noise = True,

)

settings_for_all.update(modifications)

# sigma={settings_for_all['diffusion_noise']}
name_suffix = f"{main_title}_{settings_for_all['opt']}_lr{settings_for_all['lr']}_datas={settings_for_all['dataset_size']}_{settings_for_all['nneurons'][0]}Neurons_projM={settings_for_all['use_projection_matrix']}_nlayers{len(settings_for_all['nneurons'])}"  
# _kmin={settings_for_all['k_min']}

init_exp_list = [

    dict(
        test_name= "0.0",
        diffusion_noise=0.0,
    ),
    dict(
        test_name= "0.05",
        diffusion_noise=0.05,
    ),
    dict(
        test_name= "0.1",
        diffusion_noise=0.1,
    ),
    dict(
        test_name= "0.3",
        diffusion_noise=0.3,
    ),
    dict(
        test_name= "0.8",
        diffusion_noise=0.8,
    ),
    dict(
        test_name= "1.5",
        diffusion_noise=1.5,
    ),
    dict(
        test_name= "3.0",
        diffusion_noise=3.0,
    ),
    dict(
        test_name= "10.0",
        diffusion_noise=10.0,
    ),


    
    
]

if load_all_models: 


    exp_list = []

    #for rand_seed in [10, 55, 78]:
    for e in init_exp_list: 
        temp = copy.deepcopy(e)
        temp['load_path'] = f"../scratch_link/Foundational-SDM/wandb_Logger/{temp['test_name']}{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
        exp_list.append( temp )

else: 
    exp_list = init_exp_list
    

print("Total number of experiments", len(exp_list))   


"""

dict(
        test_name= "0.001",
        activation_l1_coefficient=0.001,
    ),
    dict(
        test_name= "0.0001",
        activation_l1_coefficient=0.0001,
    ),

    dict(
        test_name= "0.00001",
        activation_l1_coefficient=0.00001,
    ),
    dict(
        test_name= "0.000001",
        activation_l1_coefficient=0.000001,
    ),
    dict(
        test_name= "0.0000001",
        activation_l1_coefficient=0.0000001,
    ),
    dict(
        test_name= "0.0",
        activation_l1_coefficient=0.0,
    ),

    

dict(
        test_name= "0.0",
        diffusion_noise=0.0,
    ),
    dict(
        test_name= "0.05",
        diffusion_noise=0.05,
    ),
    dict(
        test_name= "0.1",
        diffusion_noise=0.1,
    ),
    dict(
        test_name= "0.3",
        diffusion_noise=0.3,
    ),
    dict(
        test_name= "0.8",
        diffusion_noise=0.8,
    ),
    dict(
        test_name= "1.5",
        diffusion_noise=1.5,
    ),
    dict(
        test_name= "3.0",
        diffusion_noise=3.0,
    ),
    dict(
        test_name= "10.0",
        diffusion_noise=10.0,
    ),



    dict(
        test_name= "0.1",
        diffusion_noise=0.1,
    ),
    dict(
        test_name= "1.0",
        diffusion_noise=1.0,
    ),
    dict(
        test_name= "3.0",
        diffusion_noise=3.0,
    ),
    dict(
        test_name= "5.0",
        diffusion_noise=5.0,
    ),
    dict(
        test_name= "8.0",
        diffusion_noise=8.0,
    ),
    dict(
        test_name= "10.0",
        diffusion_noise=10.0,
    ),

"""

if __name__ == '__main__':
    print(len(exp_list))
    # copy out this script to elsewhere
    os.system(f"cp experiment_generator.py exp_commands/{main_title}_{settings_for_all['model_name']}.py")
    sys.exit(0)