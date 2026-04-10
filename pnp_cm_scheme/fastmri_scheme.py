import torch
from tqdm import tqdm
import torch.nn as nn


def get_scalings_for_boundary_condition(sigma, sigma_data=0.5, sigma_min=0.002):
    c_skip = sigma_data ** 2 / (
            (sigma - sigma_min) ** 2 + sigma_data ** 2
    )
    c_out = (
            (sigma - sigma_min)
            * sigma_data
            / (sigma ** 2 + sigma_data ** 2) ** 0.5
    )
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out, c_in


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_next_alpha(prev_alpha, gamma):
    return torch.clamp((prev_alpha * (1 + gamma)), 0, 0.9999)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(1)
    return a


def pnp_cm_fastmri_restoration(
        ATy,
        model,
        A_funcs,
        betas,
        iN,
        gamma,
        deltas,
        rhos,
        mu_0,
        T_sampling,
        config,
        classes=None,
        sigma_min=0.002,
        sigma_max=80.0,
):
    
    # Get the coefficients
    softplus  = nn.Softplus()
    penalty_param = softplus(torch.linspace(rhos[0],rhos[1],T_sampling)).to(config.device)
    momentum = (torch.tensor([mu_0]).repeat(T_sampling)/torch.arange(1,T_sampling+1)).to(config.device)
    
    xt = torch.zeros_like(ATy)
    xt_moment = torch.zeros_like(ATy)
    ut = torch.zeros_like(ATy)
    ut_moment = torch.zeros_like(ATy)
    
    t = (torch.ones(1) * (iN + 1)).to(xt.device)
    aN = compute_alpha(betas, t.long())
    alphas = [aN]
    for _ in range(T_sampling - 1):
        alphas.append(get_next_alpha(alphas[-1], gamma).reshape(1))
    
    iter_ind = -1
    
    for at in tqdm(zip(alphas)):
        
        iter_ind += 1
        
        # Get the noise level
        sigma_t_i = torch.sqrt(1 - at[0])
        sigma_cm_i = (1 + deltas[iter_ind]) * sigma_t_i
        
        sigma_t_i = torch.clamp(sigma_t_i, sigma_min, sigma_max)
        sigma_cm_i = torch.clamp(sigma_cm_i, sigma_min, sigma_max)
        
        scaling_values = get_scalings_for_boundary_condition(sigma_cm_i)
        c_skip, c_out, c_in = [append_dims(s, xt.ndim) for s in scaling_values]
        rescaled_sigma_cm = 1000 * 0.25 * torch.log(sigma_cm_i + 1e-44)
        
        xt_1 = xt.clone()
        ut_1 = ut.clone()
        
        zt = A_funcs.CG(ATy, penalty_param[iter_ind] * (xt_moment-ut_moment), penalty_param[iter_ind], cg_iter=10)
        
        z_noisy = zt + ut_moment + torch.randn_like(xt_moment) * sigma_t_i[0]
        
        if classes is None:
            et = model(c_in * z_noisy, rescaled_sigma_cm)
        else:
            et = model(c_in * z_noisy, rescaled_sigma_cm, classes)
        xt = c_out * et + c_skip * z_noisy
        xt = torch.clamp(xt, -1.0, 1.0)
              
        ut = ut_moment + (zt - xt)
        
        xt_moment = xt + momentum[iter_ind] * (xt-xt_1)
        ut_moment = ut + momentum[iter_ind] * (ut-ut_1)
    
    return xt