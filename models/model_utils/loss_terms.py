import torch 

def regularization_terms(model_params, l1_loss_weight, l2_loss_weight):  # l1 and l2 norms
    l1_sum, l2_sum = 0.0, 0.0
    for p in list(model_params):
        if l1_loss_weight is not None: 
            l1_sum += l1_loss_weight*torch.sum(torch.abs(p))
        if l2_loss_weight is not None:
            l2_sum += l2_loss_weight*torch.sum(p.pow(2.0))
    return l1_sum + l2_sum
