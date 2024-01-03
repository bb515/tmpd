from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
from jax import vmap
from tensorflow.image import ssim as tf_ssim
from tensorflow.image import psnr as tf_psnr
import jax.numpy as jnp
from jax import vmap

mse_vmap = vmap(lambda a, b: jnp.mean((a - b)**2))

device = 'cuda:0'
# loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
loss_fn_vgg = lpips.LPIPS(net='vgg')

task = 'SR'
factor = 4
sigma = 0.1
scale = 1.0  # I think that this is the step size scaling

# Ground truth
# label_root = Path(f'/media/harry/tomo/FFHQ/256_1000')
label_root = Path('/Users/ben/sync/ddrm-exp-datasets/ood_celeba/0')

# Delta must be alternative method?
# delta_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps/{scale}/recon')
# Normal must be their method?
# normal_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps+/{scale}/recon')

delta_recon_root = Path('/Users/ben/sync/ddrm-exp-datasets/ood_bedroom/0')
normal_recon_root = Path('/Users/ben/sync/ddrm-exp-datasets/ood_celeba/0')

psnr_delta_list = []
psnr_normal_list = []
ssim_delta_list = []
ssim_normal_list = []
mse_delta_list = []
mse_normal_list = []
lpips_delta_list = []
lpips_normal_list = []
# for idx in tqdm(range(150)):
for idx in tqdm(range(10)):
    # fname = str(idx).zfill(5)
    fname = str(idx)

    print(fname)
    print(label_root / f'orig_{fname}.png')
    # label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    # delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
    # normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]
    label = plt.imread(label_root / f'orig_{fname}.png')[:, :, :3]
    delta_recon = plt.imread(delta_recon_root / f'orig_{fname}.png')[:, :, :3]
    normal_recon = plt.imread(normal_recon_root / f'orig_{fname}.png')[:, :, :3]
    print(label.shape)
    print(delta_recon.shape)
    print(normal_recon.shape)

    psnr_delta = psnr(label, delta_recon)
    psnr_normal = psnr(label, normal_recon)
    ssim_delta = ssim(label, delta_recon, data_range=1.0, channel_axis=2)
    print(normal_recon.max() - normal_recon.min())
    print(delta_recon.max() - delta_recon.min())
    ssim_normal = ssim(label, normal_recon, data_range=1.0, channel_axis=2)
    mse_delta = mse(label, delta_recon)
    mse_normal = mse(label, normal_recon)
    # add outer batch fo r each image
    label = np.expand_dims(label, axis=0)
    delta_recon = np.expand_dims(delta_recon, axis=0)
    normal_recon = np.expand_dims(normal_recon, axis=0)
    _psnr_delta = tf_psnr(label, delta_recon, max_val=1.0)
    _ssim_delta = tf_ssim(label, delta_recon, max_val=1.0)
    _mse_delta = mse_vmap(label, delta_recon)
    _psnr_normal = tf_psnr(label, normal_recon, max_val=1.0)
    _ssim_normal = tf_ssim(label, normal_recon, max_val=1.0)
    _mse_normal = mse_vmap(label, normal_recon)
    print("label", np.max(label) - np.min(label))
    print("delta", np.max(delta_recon) - np.min(delta_recon), "psnr {} {}".format(psnr_delta, _psnr_delta[0]), "ssim {} {}".format(ssim_delta, _ssim_delta[0]), "mse {} {}".format(mse_delta, _mse_delta[0]))
    print("normal", np.max(normal_recon) - np.min(normal_recon), "psnr {} {}".format(psnr_normal, _psnr_normal), "ssim {} {}".format(ssim_normal, _ssim_normal), "mse {} {}".format(mse_normal, _mse_normal))

    psnr_delta_list.append(psnr_delta)
    psnr_normal_list.append(psnr_normal)
    ssim_delta_list.append(ssim_delta)
    ssim_normal_list.append(ssim_normal)
    mse_delta_list.append(mse_delta)
    mse_normal_list.append(mse_normal)

    # Pre-processing for lpips metric
    # delta_recon = torch.from_numpy(delta_recon).permute(axes).to(device)
    axes = (0, 3, 1, 2)
    delta_recon = torch.from_numpy(delta_recon).permute(axes)
    print(delta_recon.shape)
    # normal_recon = torch.from_numpy(normal_recon).permute(axes).to(device)
    normal_recon = torch.from_numpy(normal_recon).permute(axes)
    # label = torch.from_numpy(label).permute(axes).to(device)
    label = torch.from_numpy(label).permute(axes)

    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    print(delta_recon.shape)
    normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.

    delta_d = loss_fn_vgg(delta_recon, label)
    normal_d = loss_fn_vgg(normal_recon, label)
    print(delta_d)
    print(normal_d)
    assert 0

    # lpips_delta_list.append(delta_d)
    # lpips_normal_list.append(normal_d)
    lpips_delta_list.append(delta_d.detach().cpu().numpy().flatten()[0])
    lpips_normal_list.append(normal_d.detach().cpu().numpy().flatten()[0])

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
ssim_delta_avg = sum(ssim_delta_list) / len(ssim_delta_list)
mse_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)

psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
ssim_normal_avg = sum(ssim_normal_list) / len(ssim_normal_list)
mse_normal_avg = sum(mse_normal_list) / len(mse_normal_list)
lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

print(f'Delta PSNR: {psnr_delta_avg}')
print(f'Delta SSIM: {ssim_delta_avg}')
print(f'Delta MSE: {mse_delta_avg}')
print(f'Delta LPIPS: {lpips_delta_avg}')

print(f'Normal PSNR: {psnr_normal_avg}')
print(f'Normal SSIM: {ssim_normal_avg}')
print(f'Normal MSE: {mse_normal_avg}')
print(f'Normal LPIPS: {lpips_normal_avg}')
