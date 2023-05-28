import torch 
import numpy as np 

def safe_l2_norm(self,x, dims, collapse_last_dims=False):
        assert torch.isnan(x).sum() == 0, "Putting nans into the safe L2 norm!"
        if collapse_last_dims:
            l2_norm = torch.norm(x.flatten(start_dim=-2),
                                    dim=dims, keepdim=True).unsqueeze(-1)
        else:
            l2_norm = torch.norm(x, dim=dims, keepdim=True)
        z_mask = (l2_norm == 0.0) # < 0.001
        # these will give nans if divide by them.
        # I need to stop these from becoming nans.
        l2_norm[z_mask] = 1.0
        x /= l2_norm
        # mask the ones that should be nans but were just divided by 1.
        x *= (~z_mask).type(torch.float)
        return x