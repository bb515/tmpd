# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.n_iters = 950001

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  # TODO BB stuff
  # optim
  config.seed = 2023

  # sampling.cs_method = 'Boys2023ajvp'  # OOM for CelebA but not for CIFAR10, but doens't work particularly well for CIFAR10
  # sampling.cs_method = 'Boys2023avjp'  # OOM for CIFAR10
  # sampling.cs_method = 'Boys2023ajac'  # OOM for CIFAR10
  # sampling.cs_method = 'Boys2023b'  # OOM for CelebA and CIFAR10
  # sampling.cs_method = 'Song2023'  # OOM for CelebA but doesn't work (unstable) for CIFAR10
  # sampling.cs_method = 'Chung2022'  # Unstable for CIFAR10
  # sampling.cs_method = 'ProjectionKalmanFilter'
  # sampling.cs_method = 'KPDDPM'
  # sampling.cs_method = 'KGDMVP'
  sampling.cs_method = 'KPDDPM'

  # mask methods
  # sampling.cs_method = 'Song2023plus'  # Unstable at std=1.1, stable at std=1.2, stable at std=10.0
  # sampling.cs_method = 'Boys2023bvjpplus'  # Unstable, stable at std=1.2 std=10.0
  # sampling.cs_method = 'Boys2023bjvpplus'  # Unstable, stable at std=10.0
  # sampling.cs_method = 'Boys2023cplus'  # Works form noise_std = 0.003 and above. Try other methods on noise_std=0.01 and above.
  # sampling.cs_method = 'chung2022scalarplus'  # Unstable pretty much always
  # sampling.cs_method = 'chung2022plus'  # Unstable, stable at std=10.0
  # sampling.cs_method = 'KPDDPMplus'
  # sampling.cs_method = 'KPSMLDplus'
  # sampling.cs_method = 'PiGDMVPplus'
  # sampling.cs_method = 'KGDMVPplus'
  # sampling.cs_method = 'KPDDPMplus'

  sampling.noise_std = 0.001
  sampling.denoise = True  # work out what denoise_override is
  sampling.innovation = True  # this will probably be superceded
  sampling.inverse_scaler = None
  evaluate = config.eval
  evaluate.begin_ckpt = 8  # 12
  evaluate.end_ckpt = 8  # 12
  evaluate.batch_size = 1
  evaluate.pmap = False
  solver = config.solver
  # solver.outer_solver = 'eulermaruyama'
  # solver.inner_solver = None
  solver.outer_solver = 'DDIMVP'
  solver.num_outer_steps = model.num_scales
  solver.eta = 0.0  # DDIM hyperparameter

  return config