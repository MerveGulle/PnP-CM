import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from functions.fft_util import ifft2_m


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


def enforce_hermitian(X):
        # X: (B,C,H,W) complex, assumes DC at center (fftshifted layout)
        # Enforce X[k] = conj(X[-k])
        X_flip = torch.conj(torch.flip(X, dims=(-2, -1)))
        return 0.5 * (X + X_flip)


def phase_retrieval_initialization(y, oversample):
        B, C, H, W = y.shape
        r = int((oversample / 8.0) * 256)
        
        # shared phase
        g = torch.Generator(device=y.device).manual_seed(1234)
        pt = 2 * torch.pi * torch.randn((1, 1, H, W), device=y.device, dtype=y.dtype, generator=g) - torch.pi
        pt[:, :, H//2, W//2] = 0.0
        pt = pt.expand(B, C, H, W)
        
        X0 = y * torch.exp(1j * pt)
        
        # enforce Hermitian symmetry so ifft is real
        X0 = enforce_hermitian(X0)
        
        z = ifft2_m(X0).real  # should already be ~real
        
        # normalize + tanh
        mean = z.mean(dim=(2, 3), keepdim=True)
        std  = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        z = torch.tanh((z - mean) / std)
        
        return z[:, :, r:-r, r:-r]
    
    
def pnp_cm_natural_nonlinear_restoration(
        y,
        model,
        A_funcs,
        betas,
        iN,
        gamma,
        deltas,
        rhos,
        mu_0,
        classes,
        args,
        config,
        sigma_min=0.002,
        sigma_max=80.0,
):
           
    # Get the coefficients
    softplus  = nn.Softplus()
    penalty_param = softplus(torch.linspace(rhos[0],rhos[1],args.T_sampling)).to(config.device)
    momentum = (torch.tensor([mu_0]).repeat(args.T_sampling)/torch.arange(1,args.T_sampling+1)).to(config.device)
    
    xt = torch.zeros((config.sampling.batch_size,config.data.channels,config.data.image_size,config.data.image_size)).to(y.device)
    xt_moment = torch.zeros((config.sampling.batch_size,config.data.channels,config.data.image_size,config.data.image_size)).to(y.device)
    ut = torch.zeros((config.sampling.batch_size,config.data.channels,config.data.image_size,config.data.image_size)).to(y.device)
    ut_moment = torch.zeros((config.sampling.batch_size,config.data.channels,config.data.image_size,config.data.image_size)).to(y.device)
    
    if args.deg=='phase_retrieval':
        zt = phase_retrieval_initialization(y, args.phase_ret_os)
    else:
        zt = y.clone()
    
    t = (torch.ones(1) * (iN + 1)).to(xt.device)
    aN = compute_alpha(betas, t.long())
    alphas = [aN]
    for _ in range(args.T_sampling - 1):
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
        
        loss_pre = torch.tensor(0.0).to(xt.device)
        loss = torch.tensor(0.0).to(xt.device)
        
        with torch.enable_grad():            
            
            ztmp = zt.clone().requires_grad_(True)
            
            optimizer_class = getattr(optim, args.opt_type)
            optimizer = optimizer_class([ztmp], lr=float(args.opt_lr))
            
            for _ in range(args.opt_num_iter):
                res = A_funcs.A(ztmp) - y
                norm = 0.5 * (res * res).sum()
                diff = ztmp - (xt_moment - ut_moment)
                penalty = 0.5 * penalty_param[iter_ind] * torch.sum(diff * diff).abs()
                loss = norm + penalty 
                loss.backward()
                optimizer.step()
                
                if loss.item() > loss_pre.item():
                    for g in optimizer.param_groups:
                        g["lr"] = g["lr"] * args.opt_decay_rate
                
                loss_pre = loss
                    
            zt = ztmp.detach()
        del optimizer  
        
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