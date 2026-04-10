import os
import random
import lpips
import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm
from datasets import get_dataset
from pnp_cm_scheme.fastmri_scheme import pnp_cm_fastmri_restoration
from skimage.metrics import structural_similarity
from functions.mri_function import MulticoilMRI
from functions.mri_util import real_to_nchw_comp, nchw_comp_to_real, cal_PSNR, get_mvue, normalize_complex, clear

loss_fn_vgg = lpips.LPIPS(net='vgg')


def pnp_cm_fastmri_wrapper(args, config, device, betas, deltas, rhos, model, logger):
    
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
    
    # get undersampling mask
    acceleration_mask_full_path = os.path.join(args.exp, "acceleration_masks", f"{args.us_pattern}_R{args.acc_rate}_mask.npy")
    if os.path.exists(acceleration_mask_full_path):
        mask = np.load(acceleration_mask_full_path)
    else:
        from functions.mri_util import mask_generator
        mask = mask_generator(
            Nro=config.data.image_size, 
            Npe=config.data.image_size, 
            R=args.acc_rate, 
            ACS=args.acs_lines,
            mask_type=args.us_pattern
            )
    
    mask = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0)
    tvu.save_image(mask, os.path.join(args.image_folder, "acceleration_mask.png"))

    logger.info("----------------------------------------------")  
    
    # Start sampling  
    test_bar = tqdm.tqdm(test_loader, ncols=100)
    
    os.makedirs(os.path.join(args.image_folder, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.image_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(args.image_folder, "zero_filled"), exist_ok=True)
      
    psnr_arr = []
    ssim_arr = []
    
    img_ind = -1
    
    for kspace, coils, file_name in test_bar:
        
        img_ind = img_ind + 1

        ''' Loading Coil Sens. Map and Raw k-space data '''
        kspace = kspace.to(device)             # fully sampled k-space
        coils = coils.to(device)               # coil sensitivity maps
        file_name = file_name[0]                        
        
        ''' Define forward operator '''
        A_funcs = MulticoilMRI(config.data.image_size, mask, coils)
        
        ''' Sense1 Map '''
        mvue = get_mvue((kspace.clone().detach().cpu().numpy()),
                        (coils.clone().detach().cpu().numpy()))[0]
        sense1 = torch.from_numpy(mvue.astype(np.complex64)).to(device)
        norm_factor = np.max(np.abs(clear(sense1)))
        x_orig = normalize_complex(sense1)  
            
        if args.save_observed_img:
            tvu.save_image(x_orig.abs().flipud(), os.path.join(args.image_folder, "labels", f"{file_name}.png"))

        ''' Undersampling k-space '''
        y = kspace * mask / norm_factor         # normalized undersampled kspace measurement
        
        ''' Zerofilled image'''
        ATy = A_funcs.At(y)                     # normalized zero-filled image
        if args.save_observed_img:
            tvu.save_image(ATy.squeeze().abs().flipud(), os.path.join(args.image_folder, "zero_filled", f"{file_name}.png"))
            
        ATy = nchw_comp_to_real(ATy) # B, 2, H, W
        
        
        with torch.no_grad():
            x = pnp_cm_fastmri_restoration(
                ATy=ATy,
                model=model,
                A_funcs=A_funcs,
                betas=betas,
                iN=args.iN,
                gamma=args.gamma,
                deltas=deltas,
                rhos=rhos,
                mu_0=args.mu_0,
                T_sampling=args.T_sampling,
                config=config,
            )
            
        rssq_sens_map = torch.sqrt(torch.sum(coils[0]**2, dim=0)).to(device)
        x_sv = (rssq_sens_map.abs()!=0) * real_to_nchw_comp(x).squeeze()
        x_orig_sv = (rssq_sens_map.abs()!=0) * x_orig
        
        tvu.save_image(x_sv.abs().flipud(), os.path.join(args.image_folder, "samples", f"{file_name}.png"))
        
        psnr = cal_PSNR(np.abs(clear(x_orig_sv)), np.abs(clear(x_sv)))
        psnr_arr.append(psnr)
        ssim = structural_similarity(np.abs(clear(x_orig_sv)), np.abs(clear(x_sv)), data_range=1.0)
        ssim_arr.append(ssim)

        logger.info("%s, PSNR: %.2f -- SSIM: %.3f", file_name, psnr, ssim)
    
    logger.info("----------------------------------------------")
    logger.info("DATABASE RESULTS:")
    logger.info("Total Average PSNR (mean ± std): %.2f ± %.2f", np.mean(psnr_arr), np.std(psnr_arr))
    logger.info("Total Average SSIM (mean ± std): %.3f ± %.3f", np.mean(ssim_arr), np.std(ssim_arr))
    logger.info("Number of samples: %d" % (len(test_dataset)))
    logger.info("----------------------------------------------")  