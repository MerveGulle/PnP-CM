import os
import random
import lpips
import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm
from datasets import get_dataset, data_transform, inverse_data_transform
from pnp_cm_scheme.natural_nonlinear_scheme import pnp_cm_natural_nonlinear_restoration

loss_fn_vgg = lpips.LPIPS(net='vgg')



INVERSIONS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]

def adjust_color_by_corr(img, loss_fn_vgg, x_orig):
    """
    Resolve RGB sign ambiguity without diffusion model.
    img: Tensor [B, 3, H, W], values in [-1, 1]
    Returns: Tensor [B, 3, H, W]
    """
    B = img.shape[0]
    device = img.device

    def score(x):
        flat = x.view(B, 3, -1)
        flat = flat - flat.mean(dim=2, keepdim=True)
        
        return torch.squeeze(loss_fn_vgg(x.to('cpu'), x_orig.to('cpu'))).detach()

    best_score = torch.full((B,), torch.inf, device=device)
    best_img = img.clone()

    for inv in INVERSIONS:
        candidate = img.clone()
        for c, flip in enumerate(inv):
            if flip:
                candidate[:, c] *= -1

        s = score(candidate)
        s = s.view(B).to(device)
        mask = s < best_score
        best_score[mask] = s[mask]
        best_img[mask] = candidate[mask]

    return best_img

def pnp_cm_natural_nonlinear_wrapper(args, config, device, betas, deltas, rhos, model, logger):
    
    test_dataset = get_dataset(args, config)

    if args.subset_start >= 0 and args.subset_end > 0:
        assert args.subset_end > args.subset_start
        test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
    else:
        args.subset_start = 0
        args.subset_end = len(test_dataset)

    print(f'Test dataset has size {len(test_dataset)}')

    def seed_worker(worker_id):
        worker_seed = args.seed % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=config.sampling.batch_size,
        num_workers=config.data.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    # get degradation matrix
    deg = args.deg
    if deg == 'deblur_nl':
        from functions.measurements import NonlinearBlurOperator
        opt_yml_path = 'bkse/options/generate_blur/default.yml'
        A_funcs = NonlinearBlurOperator(opt_yml_path, device)
        A_funcs.blur_model.eval()

    elif deg == 'phase_retrieval':
        from functions.measurements import PhaseRetrievalOperator
        oversample = args.phase_ret_os         
        A_funcs = PhaseRetrievalOperator(oversample, device)
        import scipy.io as sio
        lpips_ref = sio.loadmat('exp/phase_retrieval_ref_img/cm_out.mat')['cm_out']
        lpips_ref = torch.from_numpy(lpips_ref).to(device)
    
    elif deg == 'jpeg':
        from functions.diff_jpeg import DiffJPEGCoding
        jpeg_qf = args.jpeg_qf
        A_funcs = DiffJPEGCoding(jpeg_qf, device)
        from functions.jpeg_torch import JPEG_ArtifactRemoval
        A_funcs_non_diff = JPEG_ArtifactRemoval(jpeg_qf, device)

    else:
        raise ValueError("degradation type not supported")

    args.sigma_y = 2 * args.sigma_y  # to account for scaling to [-1,1]

    print(f'Start from {args.subset_start}')
    idx_init = args.subset_start
    idx_so_far = args.subset_start
    avg_psnr = 0.0
    avg_lpips = 0.0
    
    logger.info("----------------------------------------------") 
    
    # Start sampling
    test_bar = tqdm.tqdm(test_loader)
        
    
    os.makedirs(os.path.join(args.image_folder, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.image_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(args.image_folder, "measurements"), exist_ok=True)
    
    img_ind = -1

    for x_orig, classes in test_bar:
        
        img_ind = img_ind + 1

        x_orig = x_orig.to(device)
        x_orig = data_transform(config, x_orig)
        
        if deg=='jpeg':
            y = A_funcs_non_diff.A(x_orig)
        else:
            y = A_funcs.A(x_orig)
        
        y = y + args.sigma_y * torch.randn_like(y).to(device)  # added noise to measurement
        
        if args.save_observed_img:
            for i in range(len(x_orig)):
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(args.image_folder, f"labels/orig_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, y[i]),
                    os.path.join(args.image_folder, f"measurements/y_{idx_so_far + i}.png")
                )               
            
        with torch.no_grad():
            x = pnp_cm_natural_nonlinear_restoration(
                y=y.detach(),
                model=model,
                A_funcs=A_funcs,
                betas=betas,
                iN=args.iN,
                gamma=args.gamma,
                deltas=deltas,
                rhos=rhos,
                mu_0=args.mu_0,
                args=args,
                config=config,
                classes=classes if config.model.class_cond else None,
            )
            
        lpips_final = torch.squeeze(loss_fn_vgg(x.to('cpu'), x_orig.to('cpu'))).detach().numpy()
        avg_lpips += lpips_final                
        

        for j in range(x.size(0)):
            if args.deg=='phase_retrieval':
                x = adjust_color_by_corr(x, loss_fn_vgg, lpips_ref)
            
            x = inverse_data_transform(config, x)
            orig = inverse_data_transform(config, x_orig[j])
            
            tvu.save_image(
                x[j], os.path.join(args.image_folder, f"samples/{idx_so_far + j}_{0}.png")
            )
            
            mse = torch.mean((x[j].to(device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr
            
            logger.info("img_ind: %d, PSNR: %.2f, LPIPS: %.3f" % (img_ind, psnr, lpips_final))
            

        idx_so_far += y.shape[0]
    
    avg_psnr = avg_psnr / (idx_so_far - idx_init)
    avg_lpips = avg_lpips / (idx_so_far - idx_init)
    
    logger.info("----------------------------------------------")
    logger.info("DATABASE RESULTS:")
    logger.info("Total Average PSNR: %.2f" % avg_psnr)
    logger.info("Total Average LPIPS: %.3f" % avg_lpips)
    logger.info("Number of samples: %d" % (idx_so_far - idx_init))
    logger.info("----------------------------------------------") 