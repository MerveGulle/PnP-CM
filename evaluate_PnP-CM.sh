#############
# CelebA-HQ #
#############
deg='sr_bicubic';       # Select the degradation: 'sr_bicubic, deblur_gauss, inpainting, deblur_nl, phase_retrieval, jpeg'
sigma_y=0.05
python main.py --deg=$deg --path_y='celeba_hq' --sigma_y=$sigma_y --config='celeba_hq_256.yml' --model_ckpt='celeba_hq/ema_0.9999432189950708_1175000.pt' --save_y


################
# LSUN Bedroom #
################
deg='deblur_gauss';       # Select the degradation: 'sr_bicubic, deblur_gauss, inpainting, deblur_nl, phase_retrieval, jpeg'
sigma_y=0.05
python main.py --deg=$deg --path_y='bedroom' --sigma_y=$sigma_y --config='lsun_bedroom_256.yml' --model_ckpt='lsun_bedroom/cd_bedroom256_lpips.pt' --save_y


#############
# FastMRI #
#############
path_y='PDFS';            # Select the MRI dataset: 'DP, PDFS'
sigma_y=0.0               # Do not change
acc_rate=4                # Set the acceleration rate as an integer
acs_lines=24              # Set the num of ACS lines as an integer
us_pattern='gaussian1d'   # Set the undersampling pattern: 'gaussian1d, equidistant'
python main.py --deg='fastmri' --path_y=$path_y --acc_rate=$acc_rate --acs_lines=$acs_lines --us_pattern=$us_pattern --sigma_y=$sigma_y --config='fastmri_320.yml' --model_ckpt=fast_mri/ema_0.9999432189950708_700000_cm_knee.pt --save_y
