"""Inpainting utilities."""
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from typing import Callable, Optional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    print(x_shape)
    x = x.reshape(-1, x_shape[-1])
    print(x.shape)
    x = x.hstack([x, jnp.flip(x, axis=1)[:, 1:-1]])[:, :, 0].reshape(x_shape)
    print(x.shape)
    assert 0
    return jnp.fft.rfft(x)
    return torch.fft.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)


def dct1_torch(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    print(x.shape)

    return torch.fft.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)


def idct1(x):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param x: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = x.shape[-1]
    return dct1(x) / (2 * (n - 1))


def idct1_torch(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.reshape(-1, N)
    v = np.hstack([x[:, ::2], jnp.flip(x[:, 1::2], axis=1)])
    Vc = jnp.fft.fft(v, axis=1)
    k = -jnp.arange(N, dtype=x.dtype)[None, :] * np.pi / (2 * N)
    W_real = jnp.cos(k)
    W_imag = jnp.sin(k)
    V = jnp.real(Vc) * W_real - jnp.imag(Vc) * W_imag
    if norm == 'ortho':
        V = V.at[:, 0].set(V[:, 0] / (jnp.sqrt(N) * 2))
        V = V.at[:, 1:].set(V[:, 1:] / (jnp.sqrt(N / 2) * 2))

    return 2 * V.reshape(x_shape)


def dct_test(x, norm):
    V_torch = np.array(dct_torch(torch.from_numpy(np.array(x)), norm))
    V = np.array(dct(x, norm))
    print(V_torch)
    print(V)
    assert np.allclose(V_torch, V)
    assert 0


def idct_torch(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def idct(x, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    N = x_shape[-1]

    X_v = x.reshape(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= jnp.sqrt(N) * 2
        X_v[:, 1:] *= jnp.sqrt(N / 2) * 2

    k = jnp.arange(x_shape[-1], dtype=x.dtype)[None, :] * np.pi / (2 * N)
    W_r = jnp.cos(k)
    W_i = jnp.sin(k)

    V_t_r = X_v
    print(-X_v.flip([1])[:, :-1].shape)
    assert 0
    V_t_i = np.hstack([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V_r = V_r.expand_dims(axis=2)
    V_i = V_i.expand_dims(axis=2)
    V = jnp.hstack(V_r, V_i)
    v = jnp.fft.irfft(V.astype(complex), n=V.shape[1], axis=1)
    x = jnp.zeros_like(v)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.reshape(x_shape)


def dct_2d_torch(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    print(X1.shape)
    assert 0
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d_torch(x, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    print(x.shape)
    x1 = idct(x, norm=norm)
    print(x1.shape)
    assert 0
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def idct_2d(x, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(x, norm=norm)
    print(x1.shape)
    assert 0
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d_torch(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d_torch(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def get_dct(in_features, _type, norm=None, bias=False):
    """Get any DCT function.

    :param in_features: size of expected input.
    :param type: which dct function in this file to use."""

    # initialise using dct function
    I = torch.eye(in_features)

    if _type == 'dct1':
        return dct1
        # weight.data = dct1(I).data.t()
    elif _type == 'idct1':
        return idct1
        # weight.data = idct1(I).data.t()
    elif _type == 'dct':
        return lambda x: dct_test(x, norm=norm)
        # return lambda x: dct(x, norm=norm)
        # TODO: does it need transposing here? because of .t()?
        # weight.data = dct(I, norm=norm).data.t()
    elif _type == 'idct':
        return lambda x: idct(x, norm=norm)
        # weight.data = idct(I, norm=norm).data.t()
    # weight.requires_grad = False # don't learn this!
    # return weight


# class LinearDCT(nn.Linear):
#     """Implement any DCT as a linear layer; in practice this executes around
#     50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
#     increase memory usage.
#     :param in_features: size of expected input
#     :param type: which dct function in this file to use"""
#     def __init__(self, in_features, type, norm=None, bias=False):
#         self.type = type
#         self.N = in_features
#         self.norm = norm
#         super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

#     def reset_parameters(self):
#         # initialise using dct function
#         I = torch.eye(self.N)
#         if self.type == 'dct1':
#             self.weight.data = dct1(I).data.t()
#         elif self.type == 'idct1':
#             self.weight.data = idct1(I).data.t()
#         elif self.type == 'dct':
#             self.weight.data = dct(I, norm=self.norm).data.t()
#         elif self.type == 'idct':
#             self.weight.data = idct(I, norm=self.norm).data.t()
#         self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    # x = torch.from_numpy(np.array(x))
    X1 = linear_layer(x)
    print(X1.shape)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)


def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


__DATASET__ = {}


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataloader(dataset: VisionDataset,
                   batch_size: int,
                   num_workers: int,
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img


def image_grid(x, image_size, num_channels):
    img = x.reshape(-1, image_size, image_size, num_channels)
    w = int(np.sqrt(img.shape[0]))
    img = img[:w**2, :, :, :]
    return img.reshape((w, w, image_size, image_size, num_channels)).transpose((0, 2, 1, 3, 4)).reshape((w * image_size, w * image_size, num_channels))


def get_asset_sample():
  dataset = 'FFHQ'
  batch_size = 4
  transform = transforms.ToTensor()
  dataset = get_dataset(dataset.lower(),
                        root='../assets/',
                        transforms=transform)
  loader = get_dataloader(dataset, batch_size=3, num_workers=0, train=False)
  ref_img = next(iter(loader))
  print(ref_img.shape)
  ref_img = ref_img.detach().cpu().numpy()[2].transpose(1, 2, 0)
  print(np.max(ref_img), np.min(ref_img))
  ref_img = np.tile(ref_img, (batch_size, 1, 1, 1))
  print(ref_img.shape)
  return ref_img


def jax_rgb2ycbcr(x):
    """Args: param x: Input signal. Assumes x in range [0, 1] and shape (N, H, W, C)."""
    # Get from [0, 1] to [0, 255]
    x = x * 255
    # plt.imshow(image_grid(jnp.uint8(x.copy()), 256, 3), interpolation='None')
    # plt.savefig("test_rgb.png")
    v = jnp.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = x.dot(v.T)
    ycbcr = ycbcr.at[:, :, :, 1:].set(ycbcr[:, :, :, 1:] + 128)
    # plt.imshow(image_grid(jnp.uint8(ycbcr.copy()), 256, 3), interpolation='None')
    # plt.savefig("test_ycbcr.png")
    return ycbcr


def jax_ycbcr2rgb(x):
    """Args: param x: Input signal. Assumes x in range [0, 1] and shape (N, H, W, C)."""
    # plt.imshow(image_grid(jnp.uint8(x.copy()), 256, 3), interpolation='None')
    # plt.savefig("test_ycbcr_.png")
    v = np.array(
      [[ 1.00000000e+00, -3.68199903e-05,  1.40198758e+00],
       [ 1.00000000e+00, -3.44113281e-01, -7.14103821e-01],
       [ 1.00000000e+00,  1.77197812e+00, -1.34583413e-04]])
    rgb = x.astype(jnp.double)
    rgb = rgb.at[:, :, :, 1:].set(rgb[:, :, :, 1:] - 128.)
    rgb = rgb.dot(v.T)
    # plt.imshow(image_grid(jnp.uint8(rgb.copy()), 256, 3), interpolation='None')
    # plt.savefig("test_rgb_.png")
    return rgb


def chroma_subsample(x):
    """
    Args: param x: Signal with shape (N, H, W, C)."""
    return x[:, :, :, 0:1], x[:, ::2, ::2, 1:]


def general_quant_matrix(qf = 10):
    q1 = jnp.array([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
    ])
    q2 = jnp.array([
        17,  18,  24,  47,  99,  99,  99,  99,
        18,  21,  26,  66,  99,  99,  99,  99,
        24,  26,  56,  99,  99,  99,  99,  99,
        47,  66,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99
    ])
    s = (5000 / qf) if qf < 50 else (200 - 2 * qf)
    q1 = torch.floor((s * q1 + 50) / 100)
    q1[q1 <= 0] = 1
    q1[q1 > 255] = 255
    q2 = torch.floor((s * q2 + 50) / 100)
    q2[q2 <= 0] = 1
    q2[q2 > 255] = 255
    return q1, q2


def quantization_matrix(qf):
    return general_quant_matrix(qf)
    # q1 = torch.tensor([[ 80,  55,  50,  80, 120, 200, 255, 255],
    #                    [ 60,  60,  70,  95, 130, 255, 255, 255],
    #                    [ 70,  65,  80, 120, 200, 255, 255, 255],
    #                    [ 70,  85, 110, 145, 255, 255, 255, 255],
    #                    [ 90, 110, 185, 255, 255, 255, 255, 255],
    #                    [120, 175, 255, 255, 255, 255, 255, 255],
    #                    [245, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255]])
    # q2 = torch.tensor([[ 85,  90, 120, 235, 255, 255, 255, 255],
    #                    [ 90, 105, 130, 255, 255, 255, 255, 255],
    #                    [120, 130, 255, 255, 255, 255, 255, 255],
    #                    [235, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255]])
    # return q1, q2


def jpeg_encode(x, qf):
    # Assume x is a batch of size (N x H x W x C)
    # [-1, 1] to [0, 255]
    # x = (x + 1) / 2 * 255
    # [0, 1] to [0, 255]
    print(x)
    n_batch, _, n_size, _ = x.shape

    x = jax_rgb2ycbcr(x)
    x_luma, x_chroma = chroma_subsample(x)
    print(x_luma.shape)
    print(x_chroma.shape)

    # github.com/google/jax/discussions/5968
    x_luma = x_luma.transpose(0, 3, 1, 2)
    x_chroma = x_chroma.transpose(0, 3, 1, 2)
    print(x_luma.shape)
    print(x_chroma.shape)

    # Assume x_luma, x_chroma is a batch of size (N x C x H x W)
    unfold_torch = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma_torch = torch.from_numpy(np.array(x_luma.copy())).to(torch.float32)
    x_chroma_torch = torch.from_numpy(np.array(x_chroma.copy())).to(torch.float32)
    x_luma_torch = unfold_torch(x_luma_torch)
    x_chroma_torch = unfold_torch(x_chroma_torch)
    print("torch")
    print(x_luma_torch.shape)
    print(x_chroma_torch.shape)
    x_luma_torch = x_luma_torch.transpose(2, 1)
    x_chroma_torch = x_chroma_torch.transpose(2, 1)
    print("torch")
    print(x_luma_torch.shape)
    print(x_chroma_torch.shape)
    x_luma = jax.lax.conv_general_dilated_patches(lhs=x_luma, filter_shape=[8, 8], padding='SAME', window_strides=[8, 8])
    x_chroma = jax.lax.conv_general_dilated_patches(lhs=x_chroma, filter_shape=[8, 8], padding='SAME', window_strides=[8, 8])
    print("jax")
    print(x_luma.shape)
    print(x_chroma.shape)
    x_luma = x_luma.reshape(x_luma.shape[0], x_luma.shape[1], -1)
    x_chroma = x_chroma.reshape(x_chroma.shape[0], x_chroma.shape[1], -1)
    print("jax")
    print(x_luma.shape)
    print(x_chroma.shape)
    x_luma = x_luma.transpose(0, 2, 1)
    x_chroma = x_chroma.transpose(0, 2, 1)
    print("jax")
    print(x_luma.shape)
    print(x_chroma.shape)
    print("compare")
    print(x_luma)
    print(x_luma_torch)
    print(x_chroma)
    print(x_chroma_torch)
    assert(np.allclose(np.array(x_luma), x_luma_torch.numpy()))
    assert(np.allclose(np.array(x_chroma), x_chroma_torch.numpy()))
    assert 0

    x_luma = x_luma.reshape(-1, 8, 8) - 128
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128
    print(x_luma.shape)
    print(x_chroma.shape)

    dct_layer = get_dct(8, 'dct', norm='ortho')
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    assert 0

    dct_layer = LinearDCT(8, 'dct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)

    x_luma = x_luma.view(-1, 1, 8, 8)
    x_chroma = x_chroma.view(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma /= q1.view(1, 8, 8)
    x_chroma /= q2.view(1, 8, 8)

    x_luma = x_luma.round()
    x_chroma = x_chroma.round()

    x_luma = x_luma.reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = x_chroma.reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    return [x_luma, x_chroma]


def jpeg_decode(x, qf):
    # Assume x[0] is a batch of size (N x 1 x H x W) (luma)
    # Assume x[1:] is a batch of size (N x 2 x H/2 x W/2) (chroma)
    x_luma, x_chroma = x
    n_batch, _, n_size, _ = x_luma.shape
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_luma = x_luma.reshape(-1, 1, 8, 8)
    x_chroma = unfold(x_chroma).transpose(2, 1)
    x_chroma = x_chroma.reshape(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma *= q1.view(1, 8, 8)
    x_chroma *= q2.view(1, 8, 8)

    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)

    dct_layer = LinearDCT(8, 'idct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)

    x_luma = (x_luma + 128).reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = (x_chroma + 128).reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    x_chroma_repeated = torch.zeros(n_batch, 2, n_size, n_size, device = x_luma.device)
    x_chroma_repeated[:, :, 0::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 0::2, 1::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 1::2] = x_chroma

    x = torch.cat([x_luma, x_chroma_repeated], dim=1)

    x = torch_ycbcr2rgb(x)

    # [0, 255] to [-1, 1]
    x = x / 255 * 2 - 1

    return x


def quantization_encode(x, qf):
    qf = 32
    #to int
    x = (x + 1) / 2
    x = x * 255
    x = x.int()
    # quantize
    x = x // qf
    #to float
    x = x.float()
    x = x / (255/qf)
    x = (x * 2) - 1
    return x


def quantization_decode(x, qf):
    return x


x = jnp.array(get_asset_sample())
y = jax_rgb2ycbcr(x)
jax_ycbcr2rgb(y)
qf = 10.
jpeg_encode(x, qf)
