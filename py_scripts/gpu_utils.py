import subprocess
import pandas as pd 

def get_free_gpus(mem_threshold, activity_threshold=80, use_all=False, total_gpus=4):

    # returns a list of the GPU indices along with a string depending on what it is best to pass them into. 
    # eg. os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_list = out_str[0].decode("utf-8").split('\n')

    gpu_ind = 0
    gpu_stats = dict()
    prev_item = None
    for item in out_list:
        if 'Used                              :' in item and '256 MiB' not in prev_item:
            gpu_memory = int(item.split(':')[-1].split('MiB')[0])
        if 'Gpu                               :' in item:
            gpu_act = int(item.split(':')[-1].split('%')[0])
            gpu_stats[gpu_ind] = dict(activity=gpu_act, memory_used=gpu_memory)
            gpu_ind+=1
        prev_item = item

    gpus_to_use = []
    for k, v in gpu_stats.items():
        if v['activity']<activity_threshold and v['memory_used']<mem_threshold:
            gpus_to_use.append(k)
    print(gpu_stats)

    gpu_stats = pd.DataFrame(gpu_stats).T
    try: 
        display(gpu_stats)
    except: 
        print(gpu_stats)

    if not use_all and len(gpus_to_use) == total_gpus: 
        print("Disabled use of all GPUS! Removing one.")
        gpus_to_use.pop()

    # make a string of gpus to use
    gpu_str = ''
    for g in gpus_to_use:
        gpu_str+=str(g)+', '

    

    print('IDs of GPUs to use:', gpus_to_use)

    return gpus_to_use, gpu_str


    '''def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values'''