import os
import lpips
import numpy as np
import torch
from guided_diffusion.script_util import create_model

loss_fn_vgg = lpips.LPIPS(net='vgg')


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class PnP_CM_Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        
        # Set deltas
        if args.deltas == "":
            deltas = [0.1] * args.T_sampling
        else:
            deltas = [float(x.strip()) for x in args.deltas.split(',')]
            if len(deltas) != args.T_sampling:
                raise ValueError(
                    f"deltas length must be equal to T_sampling. Got {len(deltas)} deltas, but {args.T_sampling} sampling steps.")
        self.deltas = deltas
        
        # Set rhos
        if args.rhos == "":
            rhos = [-4.0,4.0]
        else:
            rhos = [float(x.strip()) for x in args.rhos.split(',')]
            if len(rhos) != 2:
                raise ValueError(
                    f"rhos length must be equal to 2, but got {len(rhos)} rhos.")
        self.rhos = rhos
        
    def sample(self, logger):
        config_dict = vars(self.config.model)
        config_dict.update({"image_size": self.config.data.image_size})
        model = create_model(**config_dict)
        ckpt = os.path.join(self.args.exp, f"logs/{self.args.model_ckpt}")
        if not os.path.exists(ckpt):
            raise ValueError(f"Model ckpt not found in: {ckpt}. Please refer to https://github.com/MerveGulle/PnP-CM.git"
                             f"to see where the models that were used in the paper can be found.")

        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        if self.config.model.use_fp16:
            model.convert_to_fp16()
        model.eval()
        model = torch.nn.DataParallel(model, device_ids=[self.args.device_ids])
        
        if self.args.deg in ['sr_bicubic', 'deblur_gauss', 'inpainting']:
            from wrappers.natural_linear_wrapper import pnp_cm_natural_linear_wrapper
            print('Running PnP-CM.',
                f'Dataset: {self.args.path_y}',
                f'Task: {self.args.deg}.',
                f'Noise level: {self.args.sigma_y}.'
                )
            pnp_cm_natural_linear_wrapper(self.args, self.config, self.device, self.betas, self.deltas, self.rhos, model, logger)
        
        elif self.args.deg in ['jpeg', 'deblur_nl', 'phase_retrieval']:
            from wrappers.natural_nonlinear_wrapper import pnp_cm_natural_nonlinear_wrapper
            print('Running PnP-CM.',
                f'Dataset: {self.args.path_y}',
                f'Task: {self.args.deg}.',
                f'Noise level: {self.args.sigma_y}.'
                )
            pnp_cm_natural_nonlinear_wrapper(self.args, self.config, self.device, self.betas, self.deltas, self.rhos, model, logger)
        
        elif self.args.deg=='fastmri':
            from wrappers.fastmri_wrapper import pnp_cm_fastmri_wrapper
            print('Running PnP-CM.',
                f'Dataset: {self.args.path_y}',
                f'Task: {self.args.deg}.',
                f'Acceleration rate: {self.args.acc_rate}.'
                f'Undersampling pattern: {self.args.us_pattern}.'
                )
            pnp_cm_fastmri_wrapper(self.args, self.config, self.device, self.betas, self.deltas, self.rhos, model, logger)
        
        else:
            raise ValueError("degradation type not supported")
        