# PnP-CM: Consistency Models as Plug-and-Play Priors for Inverse Problems (CVPR 2026) #

[Merve Gulle](https://scholar.google.com/citations?user=Pmu-yJYAAAAJ&hl=en), [Junno Yun](https://scholar.google.com/citations?user=Ou4ff9kAAAAJ&hl=en&oi=ao), [Yasar Utku Alcalar](https://scholar.google.com/citations?user=9N2YMjEAAAAJ&hl=en&oi=ao), [Mehmet Akcakaya](https://scholar.google.com/citations?user=x-q3XC4AAAAJ&hl=en&oi=ao), University of Minnesota

---

# Installation #

### 1. Clone This Repository

```bash
git clone https://github.com/MerveGulle/PnP-CM.git
```

### 2. Create Conda Environment and Install Requirements

```bash
conda create -n {env_name} python=3.12
conda activate {env_name}

cd CM-RED
pip install requirements.txt
```

### 3. Pre-trained CM Models

We provide pre-trained CM models for the **CelebA-HQ** and **fastMRI knee** datasets. \
For **LSUN Bedroom**, we used the official pre-trained Consistency Models provided by OpenAI in the [CM repo](https://github.com/openai/consistency_models?tab=readme-ov-file#pre-trained-models).\

#### CelebA-HQ
Pre-trained CM model is available at the following [link](https://www.dropbox.com/scl/fo/l5q06udyq1zbg2rhjbvvm/AFSAMaZbHmJNG1Nd1qyJ-Ko?rlkey=h2np6dpba8tnnv3pc66ew5o7x&dl=0). \
Please download and place the pre-trained models under: ./exp/logs/celeba_hq/
```bash
./exp/logs/celeba_hq/
```

#### LSUN Bedroom
Pre-trained CM model is available at the following [link]([https://www.dropbox.com/scl/fo/l5q06udyq1zbg2rhjbvvm/AFSAMaZbHmJNG1Nd1qyJ-Ko?rlkey=h2np6dpba8tnnv3pc66ew5o7x&dl=0](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt)). \
Please download and place the pre-trained models under: ./exp/logs/lsun_bedroom/
```bash
./exp/logs/lsun_bedroom/
```

#### Fast MRI (knee)
Pre-trained CM model is available at the following [link](https://www.dropbox.com/scl/fo/l5q06udyq1zbg2rhjbvvm/AFSAMaZbHmJNG1Nd1qyJ-Ko?rlkey=h2np6dpba8tnnv3pc66ew5o7x&dl=0). \
Please download and place the pre-trained models under: ./exp/logs/fast_mri/
```bash
./exp/logs/fast_mri/
```


### 4. Datasets

#### CelebA-HQ

#### LSUN Bedroom

#### Fast MRI knee
Please download the **fastMRI** dataset from [fastMRI](https://fastmri.med.nyu.edu/) after agreeing to the data use agreement.

We use the `knee_multicoil_val` validation sets for evaluation. 

Coil sensitivity maps are generated using the `sigpy.mri.app.EspiritCalib` function.

The preprocessed dataset should be placed under:

```bash
./exp/datasets/fastMRI/
├── PD
├── PDFS
```

Make sure the preprocessed files are in **.mat format** and contain the following keys:

```python
# k-space data
kspace  # shape: (C, H, W)

# coil sensitivity maps
coils   # shape: (C, H, W)
```

`./datasets/fast_mri.py` loads the raw k-space data and the corresponding coil sensitivity maps.


## Quick Start
Use the following commands to generate PnP-CM results:

#### CelebA-HQ
```bash
python main.py --deg={deg} --path_y='celeba_hq' --sigma_y=0.05 --config='celeba_hq_256.yml' --model_ckpt='celeba_hq/ema_0.9999432189950708_1175000.pt' --save_y
```
#### LSUN Bedroom
```bash
python main.py --deg={deg} --path_y='bedroom' --sigma_y=$sigma_y --config='lsun_bedroom_256.yml' --model_ckpt='lsun_bedroom/cd_bedroom256_lpips.pt' --save_y
```
#### Fast MRI knee
```bash
python main.py --deg='fastmri' --path_y={path_y} --acc_rate=4 --acs_lines=24 --us_pattern=gaussian1d --sigma_y=0.0 --config='fastmri_320.yml' --model_ckpt=fast_mri/ema_0.9999432189950708_700000_cm_knee.pt --save_y
```
