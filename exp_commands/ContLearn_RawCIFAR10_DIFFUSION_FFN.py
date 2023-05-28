import os
import sys
import copy
import torch 

settings_for_all = dict(
    epochs_to_train_for = 1000,
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

main_title = "ContLearn_RawCIFAR10"

modifications = dict(

    continual_learning = True, 
    classification = True, 
    split_random_seed = None, #3,15,27,97
    use_bias_output = False, 
    epochs_per_cl_task = 50,  
    epochs_to_train_for = 250,

    dataset_name = "CIFAR10",
    #opt='SparseAdam',
    #lr=0.00001,

    #nneurons=[1_000],
    #noise_type = "expo"

    #act_func = torch.nn.Sigmoid(),

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


epoch_and_step = "epoch=999-step=49000.ckpt"
exp_name_path = "Interpretable_NoCosine_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1" #"Baseline_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"
exp_list = [

    dict(
        test_name= "0.0",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/0.0{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "0.05",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/0.05{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "0.1",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/0.1{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "0.3",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/0.3{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "0.8",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/0.8{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "1.5",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/1.5{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "3.0",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/3.0{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    dict(
        test_name= "10.0",
        load_path=f"../scratch_link/Foundational-SDM/wandb_Logger/10.0{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
    ),
    
]

"""

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
        test_name= "0.0",
        adversarial_max_distortion=0.0,
    ),
    dict(
        test_name= "0.01",
        adversarial_max_distortion=0.01,
    ),
    dict(
        test_name= "0.03",
        adversarial_max_distortion=0.03,
    ),
    dict(
        test_name= "0.06",
        adversarial_max_distortion=0.06,
    ),
    dict(
        test_name= "0.1",
        adversarial_max_distortion=0.1,
    ),
    dict(
        test_name= "0.15",
        adversarial_max_distortion=0.15,
    ),
    dict(
        test_name= "0.3",
        adversarial_max_distortion=0.3,
    ),
    dict(
        test_name= "0.6",
        adversarial_max_distortion=0.6,
    ),


dict(
        test_name= "Baseline",
        model_name= "DIFFUSION_FFN",
    ),

    dict(
        test_name= "Baseline_SharedBias",
        model_name= "DIFFUSION_FFN",
        use_shared_bias = True, 
    ),

    dict(
        test_name= "sdm_norm_all_SharedB",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=True,
        k_approach = "LEARN_K_SHARED_BIAS",
    ),

    dict(
        test_name= "sdm_norm_addr_and_vals_SharedB",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=False,
        k_approach = "LEARN_K_SHARED_BIAS",
    ),

    dict(
        test_name= "sdm_norm_address_SharedB",
        norm_addresses=True, 
        norm_values = False, 
        norm_activations=False,
        k_approach = "LEARN_K_SHARED_BIAS",
    ),

    dict(
        test_name= "sdm_norm_none_SharedB",
        norm_addresses=False, 
        norm_values = False, 
        norm_activations=False,
        k_approach = "LEARN_K_SHARED_BIAS",
    ),

    dict(
        test_name= "sdm_norm_all_KSig",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=True,
        k_approach = "LEARN_K_SIGMOID",
    ),

    dict(
        test_name= "sdm_norm_addr_and_vals_KSig",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=False,
        k_approach = "LEARN_K_SIGMOID",
    ),

    dict(
        test_name= "sdm_norm_address_KSig",
        norm_addresses=True, 
        norm_values = False, 
        norm_activations=False,
        k_approach = "LEARN_K_SIGMOID",
    ),
    dict(
        test_name= "sdm_norm_none_KSig",
        norm_addresses=False, 
        norm_values = False, 
        norm_activations=False,
        k_approach = "LEARN_K_SIGMOID",
    ),



    dict(
        
        test_name= "100N_ffn",
        nneurons=[100],
    ),
    dict(
        
        test_name= "1_000N_ffn",
        nneurons=[1000],
    ),
    dict(
        
        test_name= "10_000N_ffn",
        nneurons=[10000],
    ),
    dict(
        
        test_name= "100_000N_ffn",
        nneurons=[100000],
    ),
    =============================

    ---------------

    dict(
        
        diffusion_noise=0.0,
        test_name= "0sigma_ffn",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.0,
        test_name= "0sigma_sdm",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.0,
        test_name= "0sigma_sdm_nol2",
        norm_addresses=False, 
        norm_values = False, 
        norm_activations=False,
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.0,
        test_name= "0sigma_sdm_flat",
        k_approach = "FLAT_MASK",
        
    ),

    dict(
        
        
        test_name= "ffn",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm",
    ),

    dict(
        
        diffusion_noise=0.5,
        test_name= "0.5sigma_ffn",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        diffusion_noise=0.5,
        test_name= "0.5sigma_sdm",
    ),


#####

    
-----



dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=100,
        test_name= "sdm_flat_k=100",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=50,
        test_name= "sdm_flat_k=500",
        k_approach = "FLAT_MASK",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=20,
        test_name= "sdm_flat_k=50",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=1000,
        test_name= "sdm_flat_k=2000",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=3000,
        test_name= "sdm_flat_k=5000",
        k_approach = "FLAT_MASK",
    ),
    dict(
        model_name= "DIFFUSION_SDM,
        
        k_min=8000,
        test_name= "sdm_flat_k=8000",
        k_approach = "FLAT_MASK",
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm_norm_all",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=True,
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm_norm_addr_and_vals",
        norm_addresses=True, 
        norm_values = True, 
        norm_activations=False,
    ),

    dict(
        model_name= "DIFFUSION_SDM,
        
        test_name= "sdm_norm_address",
        norm_addresses=True, 
        norm_values = False, 
        norm_activations=False,
    ),


    ----

    dict(
        
        test_name= "0.1_ffn",
        diffusion_noise=0.1,
    ),
    dict(
        
        test_name= "0.3_ffn",
        diffusion_noise=0.3,
    ),
    dict(
        
        test_name= "0.6_ffn",
        diffusion_noise=0.6,
    ),
    
    dict(
        
        test_name= "1.0_ffn",
        diffusion_noise=1.0,
    ),
    dict(
        
        test_name= "1.5_ffn",
        diffusion_noise=1.5,
    ),

"""

if __name__ == '__main__':
    print(len(exp_list))
    # copy out this script to elsewhere
    os.system(f"cp experiment_generator.py exp_commands/{main_title}_{settings_for_all['model_name']}.py")
    sys.exit(0)