'''
Based on: https://github.com/DPS2022/diffusion-posterior-sampling
This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.
'''

from abc import ABC, abstractmethod
import yaml
from torch.nn import functional as F
from torchvision import torch
import os
from functions.fft_util import fft2_m, ifft2_m
import numpy as np


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        mod_gen = torch.randint(0, 8, size=(256, 256), device=self.device)
        self.modulation = -(mod_gen == 0).long() + (mod_gen == 7).long()
        
    def modulate(self, data, **kwargs):
        return data * self.modulation.expand_as(data)

    def forward(self, data, **kwargs):
        padded = F.pad(self.modulate(data), (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def fft(self, data, **kwargs):
        padded = F.pad(self.modulate(data), (self.pad, self.pad, self.pad, self.pad))
        fft = fft2_m(padded)
        return fft

    def ifft(self, f, **kwargs):
        padded = ifft2_m(f)
        ifft_f = padded[..., self.pad:-self.pad, self.pad:-self.pad]
        return ifft_f
    
    def A(self, data, epsilon=1e-6, **kwargs):
        padded = F.pad(self.modulate(data), (self.pad, self.pad, self.pad, self.pad))
        Fx = fft2_m(padded)
        amplitude = (Fx.real**2 + Fx.imag**2 + epsilon**2).sqrt()
        return amplitude
    
    
@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = "bkse/" + opt["pretrained"]
            model_path = os.path.join(work_dir, model_path)
        blur_model = KernelWizard(opt)
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        # torch.manual_seed(8)
        # random_kernel = torch.randn(data.shape[0], 512, 2, 2).to(self.device) * 1.2
        random_kernel = torch.from_numpy(np.load("exp/nonlinear_blurring_kernels/random_kernel_seed_8.npy")).to(data.device)
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0)
        return blurred
    
    def A(self, data, **kwargs):
        return self.forward(data)
