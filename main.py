import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # General Setting
    parser.add_argument("--config", type=str, default='celeba_hq_256.yml', help="Path to the config file") # e.g.,  lsun_bedroom_256.yml, celeba_hq_256.yml, fastmri_320.yml
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument("--device_ids", type=int, default=0, help="cuda=?")
    
    # Model
    parser.add_argument("--model_ckpt", type=str, default='celeba_hq/ema_0.9999432189950708_1175000.pt', help="Name of the model checkpoint") # e.g.,  lsun_bedroom/cd_bedroom256_lpips.pt, celeba_hq/ema_0.9999432189950708_1175000.pt, fast_mri/ema_0.9999432189950708_700000_cm_knee.pt
    
    # Save / Log
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--save_y", dest="save_observed_img", action="store_true")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_false", help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--path_y", type=str, default='celeba_hq', help="Path of the test dataset.") # e.g.,  celeba_hq, bedroom, PD, PDFS
    
    # Degradation
    parser.add_argument("--deg", type=str, default='deblur_gauss', help="Degradation - sr_bicubic, deblur_gauss, inpainting, deblur_nl") # e.g., sr_bicubic, deblur_gauss, inpainting, deblur_nl, phase_retrieval, jpeg, fastmri
    parser.add_argument("--sigma_y", type=float, default=0.05, help="sigma_y")
    parser.add_argument("--operator_imp", type=str, default="SVD", help="SVD | FFT")
    parser.add_argument("--deg_scale", type=float, default=4.0, help="deg_scale for SR Bicubic")
    parser.add_argument("--inpainting_mask_path", type=str, default="random_70_mask.npy", help="Path to mask NPY path for inpainting")
    parser.add_argument("--phase_ret_os", type=float, default=4.0, help="Phase retrieval oversampling rate")
    parser.add_argument("--jpeg_qf", type=int, default=5, help="JPEG restoration quantization factor")
    parser.add_argument("--acc_rate", type=int, default=4, help="Fast MRI acceleration rate")
    parser.add_argument("--acs_lines", type=int, default=24, help="Fast MRI number of ACS lines")
    parser.add_argument("--us_pattern", type=str, default="gaussian1d", help="Fast MRI undersampling pattern: equidistant, gaussian1d")
    
    # Data
    parser.add_argument('--subset_start', type=int, default=-1)
    parser.add_argument('--subset_end', type=int, default=-1)
    
    # Hyperparameters
    parser.add_argument("--iN", type=int, default=150, help="iN hyperparameter, initial noise level")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma hyperparameter, noise level decay rate")
    parser.add_argument("--deltas", type=str, default="", help="A comma separated list of the delta hyperparameters, sigma_cm[n]=sigma[n]*(1+delta[n])")
    parser.add_argument("--rhos", type=str, default="", help="A comma separated list of the rho hyperparameters, penalty parameters")
    parser.add_argument("--mu_0", type=float, default=0.0, help="Mu (initial momentum) hyperparameter")
    parser.add_argument("--T_sampling", type=int, default=4, help="Number of iterations / NFEs")
    
    # DC optimizer for nonlinear problems
    parser.add_argument("--opt_type", type=str, default='SGD', help="Adam, SGD")
    parser.add_argument("--opt_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--opt_num_iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--opt_decay_rate", type=float, default=0.8, help="Decay rate of the learning rate")
    
    args = parser.parse_args()            
    
    # parse predefined args from YAML
    if args.deg=="fastmri":
        args_file = os.path.join("task_specific_args", f"{args.deg}_{args.path_y}_R{args.acc_rate}_ACS{args.acs_lines}_{args.us_pattern}.yml")
    else:
        args_file = os.path.join("task_specific_args", f"{args.path_y}_{args.deg}_sigma_y_{args.sigma_y}.yml")
    if os.path.exists(args_file):
        with open(args_file, "r") as f:
            hyperparameters = yaml.safe_load(f)
        
        for section, params in hyperparameters.items():
            print("Section:", section)
            
            if isinstance(params, dict):
                for key, value in params.items():
                    setattr(args, key, value)


    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)


    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))


    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)
    
    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    if args.deg=="fastmri":
        image_folder = f"PnP_CM-{args.deg}-{args.path_y}-R{args.acc_rate}_ACS{args.acs_lines}_{args.us_pattern}-N_{args.T_sampling}-iN_{args.iN}-gamma_{args.gamma}-deltas_{args.deltas}-rhos_{args.rhos}-mu_{args.mu_0}"
    else:
        image_folder = f"PnP_CM-{args.path_y}-{args.deg}-sigma_y_{args.sigma_y}-N_{args.T_sampling}-iN_{args.iN}-gamma_{args.gamma}-deltas_{args.deltas}-rhos_{args.rhos}-mu_{args.mu_0}"
    args.image_folder = os.path.join(
        args.exp, "image_samples", image_folder
    )
    
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)


    log_path = os.path.join(args.image_folder, '0_logs.log')
    fh = logging.FileHandler(log_path)  # , mode='a')
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)


    # add device
    device = torch.device(f"cuda:{args.device_ids}") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device


    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, logger = parse_args_and_config()
    
    try:
        from guided_diffusion.diffusion_pnp_cm import PnP_CM_Diffusion
        runner = PnP_CM_Diffusion(args, config, config.device)
        runner.sample(logger)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())