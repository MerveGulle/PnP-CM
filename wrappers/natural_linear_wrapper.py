import os
import random
import lpips
import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm
from datasets import get_dataset, data_transform, inverse_data_transform
from pnp_cm_scheme.natural_linear_scheme import pnp_cm_natural_linear_restoration
from functions.median_inpainting import median_inpainting

loss_fn_vgg = lpips.LPIPS(net='vgg')


def pnp_cm_natural_linear_wrapper(args, config, device, betas, deltas, rhos, model, logger):
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
        shuffle=False,
    )

    # get degradation matrix
    deg = args.deg
    if deg == 'sr_bicubic':
        factor = int(args.deg_scale)
        
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)

        if args.operator_imp == 'SVD':
            from functions.svd_operators import SRConv
            A_funcs = SRConv(kernel / kernel.sum(), config.data.channels, config.data.image_size, device,
                                stride=factor)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Superres_fft, prepare_cubic_filter
            k = prepare_cubic_filter(1 / factor)
            kernel = torch.from_numpy(k).float().to(device)
            A_funcs = Superres_fft(kernel / kernel.sum(), config.data.channels, config.data.image_size,
                                    device, stride=factor)
        else:
            raise NotImplementedError()

    elif deg == 'deblur_gauss':
        sigma = 10  # better make argument for kernel type
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(
            device)  # clip it as in DDRM/DDNM code, but it makes more sense to use lower sigma with the line below
        
        # kernel = torch.Tensor([pdf(ii) for ii in range(-30,31,1)]).to(device)
        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size,
                                    device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(kernel / kernel.sum(), config.data.channels, config.data.image_size,
                                        device)
        else:
            raise NotImplementedError()
    
    elif deg == 'inpainting':
        from functions.svd_operators import Inpainting
        inpainting_mask_full_path = os.path.join(args.exp, "inpainting_masks", args.inpainting_mask_path)
        if not os.path.exists(inpainting_mask_full_path):
            raise ValueError(f"Could not find inpainting mask in path: {inpainting_mask_full_path}."
                                f"Please set the correct path using the --inpainting_mask_path flag.")
        loaded = np.load(inpainting_mask_full_path)
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)
    
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
        
        y = A_funcs.A(x_orig)
        
        y = y + args.sigma_y * torch.randn_like(y).to(device)  # added noise to measurement
        
        b, hwc = y.size()
        if 'color' in deg:
            hw = hwc / 1
            h = w = int(hw ** 0.5)
            y = y.reshape((b, 1, h, w))
        elif 'inp' in deg or 'cs' in deg:
            pass
        else:
            hw = hwc / 3
            h = w = int(hw ** 0.5)
            y = y.reshape((b, 3, h, w))

        y = y.reshape((b, hwc))

        ATy = A_funcs.At(y).view(y.shape[0],
                                 config.data.channels,
                                 config.data.image_size,
                                 config.data.image_size)
        
        if 'inpainting' in args.deg:
            Apy = A_funcs.A_pinv_add_eta(y).view(y.shape[0],
                                                 config.data.channels,
                                                 config.data.image_size,
                                                 config.data.image_size)
        if args.save_observed_img:
            for i in range(len(x_orig)):
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(args.image_folder, f"labels/orig_{idx_so_far + i}.png")
                )
                if 'inpainting' in deg:
                    tvu.save_image(
                        inverse_data_transform(config, Apy[i]),
                        os.path.join(args.image_folder, f"measurements/y_{idx_so_far + i}.png")
                    )
                else:
                    tvu.save_image(
                        inverse_data_transform(config, y[i].reshape((3, h, w))),
                        os.path.join(args.image_folder, f"measurements/y_{idx_so_far + i}.png")
                    )               
            
        if 'inpainting' in args.deg:
            x_init = median_inpainting(
                Apy,
                mask,
                config.data.channels,
                config.data.image_size,
            ).view(
                y.shape[0], config.data.channels,
                config.data.image_size,
                config.data.image_size,
            )
        else:
            x_init = torch.zeros_like(x_orig).detach()
        
        
        with torch.no_grad():
            x = pnp_cm_natural_linear_restoration(
                ATy=ATy,
                x_init=x_init,
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
                classes=classes if config.model.class_cond else None,
            )
            
        lpips_final = torch.squeeze(loss_fn_vgg(x.to('cpu'), x_orig.to('cpu'))).detach().numpy()
        avg_lpips += lpips_final                
        

        for j in range(x.size(0)):
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