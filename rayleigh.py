"""Rayleigh example."""
from absl import app, flags
from ml_collections.config_flags import config_flags
import jax
from jax import vmap, grad, jit
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import Partial as partial
import numpyro.distributions as dist
import pandas as pd
from tqdm import trange
import torch
from torch import eye, randn_like, vstack, manual_seed
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
from diffusionjax.plot import plot_heatmap
from diffusionjax.utils import get_sampler, get_times
from diffusionjax.run_lib import get_ddim_chain
from diffusionjax.solvers import EulerMaruyama
import diffusionjax.sde as sde_lib
from tmpd.samplers import get_cs_sampler, get_stsl_sampler
from tmpd.plot import plot_single_image, plot_image
import numpy as np
import logging
import ot
from scipy.stats import wasserstein_distance
import time


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", "./configs/rayleigh.py", "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "./workdir", "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
logger = logging.getLogger(__name__)


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
  # projections = torch.randn(size=(n_slices, dist_1.shape[1])).to(dist_1.device)
  projections = torch.randn(size=(n_slices, dist_1.shape[1]))
  projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
  dist_1_projected = (projections @ dist_1.T)
  dist_2_projected = (projections @ dist_2.T)
  return np.mean([wasserstein_distance(u_values=d1.cpu().numpy(), v_values=d2.cpu().numpy()) for d1, d2 in zip(dist_1_projected, dist_2_projected)])


def ot_sliced_wasserstein(seed, dist_1, dist_2, n_slices=100):
  return ot.sliced_wasserstein_distance(dist_1, dist_2, n_projections=n_slices, seed=seed)


def main(argv):
  workdir = FLAGS.workdir
  config = FLAGS.config
  jax.default_device = jax.devices()[0]
  # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
  # ... they must be all the same model of device for pmap to work
  num_devices =  int(jax.local_device_count()) if config.training.pmap else 1
  color_posterior = '#a2c4c9'
  color_algorithm = '#ff7878'

  # Setup SDE
  if config.training.sde.lower()=='vpsde':
    from diffusionjax.utils import get_linear_beta_function
    beta, log_mean_coeff = get_linear_beta_function(config.model.beta_min, config.model.beta_max)
    sde = sde_lib.VP(beta, log_mean_coeff)
  elif config.training.sde.lower()=='vesde':
    from diffusionjax.utils import get_sigma_function
    sigma = get_sigma_function(config.model.sigma_min, config.model.sigma_max)
    sde = sde_lib.VE(sigma)
  else:
    raise NotImplementedError(f"SDE {config.training.SDE} unknown.")

  rng = random.PRNGKey(2023)
  C = jnp.eye(2)

  def nabla_log_pt(x, t):
    r"""
    Args:
        x: One location in $\mathbb{R}^{image_size**2}$
        t: time
    Returns:
        The true log density.
        .. math::
            p_{t}(x)
    """
    x_shape = x.shape
    v_t = sde.variance(t)
    m_t = sde.mean_coeff(t)
    x = x.flatten()
    score = -jnp.linalg.solve(m_t**2 * C + v_t * jnp.eye(x.shape[0]), x)
    return score.reshape(x_shape)

  true_score = jit(vmap(nabla_log_pt, in_axes=(0, 0), out_axes=(0)))
  model = lambda x, t: -jnp.sqrt(sde.variance(t)) * true_score

  # Prior sampling
  p_samples = random.multivariate_normal(rng, mean=jnp.zeros(config.data.image_size,),
      cov=C, shape=(config.eval.batch_size,))
  C_emp = jnp.cov(p_samples[:, :].T)
  m_emp = jnp.mean(p_samples[:, :].T, axis=1)
  corr_emp = jnp.corrcoef(p_samples[:, :].T)
  plot_heatmap(samples=p_samples[:, [0, 1]], area_bounds=[-3., 3.], fname="target_prior_heatmap")
  print(p_samples.shape)
  p_samples = p_samples.reshape(config.eval.batch_size, config.data.image_size)
  delta_t_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
  delta_t_var = jnp.linalg.norm(jnp.diag(C) - jnp.diag(C_emp)) / config.data.image_size
  delta_t_mean = jnp.linalg.norm(m_emp) / config.data.image_size
  delta_t_corr = jnp.linalg.norm(C - corr_emp) / config.data.image_size
  logging.info("analytic_prior delta_mean={}, delta_var={}, delta_cov={}".format(
      delta_t_mean, delta_t_var, delta_t_cov))

  # Running the reverse SDE with the true score
  ts, _ = get_times(num_steps=config.solver.num_outer_steps)
  solver = EulerMaruyama(sde.reverse(true_score), ts)

  sampler= get_sampler((config.eval.batch_size//num_devices, config.data.image_size), solver)
  if config.eval.pmap:
      sampler = jax.pmap(sampler, axis_name='batch')
      rng, *sample_rng = random.split(rng, 1 + num_devices)
      sample_rng = jnp.asarray(sample_rng)
  else:
      rng, sample_rng = random.split(rng, 1 + num_devices)
  q_samples, _ = sampler(sample_rng)
  q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size)

  C_emp = jnp.cov(q_samples[:, :].T)
  m_emp = jnp.mean(q_samples[:, :].T, axis=1)
  corr_emp = jnp.corrcoef(q_samples[:, :].T)
  delta_corr = jnp.linalg.norm(C - corr_emp) / config.data.image_size
  delta_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
  delta_mean = jnp.linalg.norm(m_emp) / config.data.image_size
  plot_heatmap(samples=q_samples[:, [0, 1]], area_bounds=[-3., 3.], fname="diffusion_prior_heatmap")

  q_samples = q_samples.reshape(config.eval.batch_size, config.data.image_size)
  delta_cov = jnp.linalg.norm(C - C_emp) / config.data.image_size
  delta_var = jnp.linalg.norm(jnp.diag(C) - jnp.diag(C_emp)) / config.data.image_size
  delta_mean = jnp.linalg.norm(m_emp) / config.data.image_size
  logging.info("diffusion_prior delta_mean={}, delta_var={}, delta_cov={}".format(
      delta_mean, delta_var, delta_cov))

  logging.info(delta_t_corr)  # a value of 0.05 (for 512 samples) are indistinguisable from
  # true samples due to emprical covariance error
  # but it is possible to get a value as los as 0.005 from many more true samples
  logging.info(delta_corr)  # a value of 0.1 are good samples

  # def observation_map(x):
  #   return jnp.linalg.norm(x).reshape(1)
    # return jnp.array([jnp.linalg.norm(x), jnp.arctan(x[0] / x[1])])

  def observation_map(x):
    x = x.flatten()
    return jnp.array([jnp.sqrt(x[0]**2)])  # Condition on modulus of one coordinate
    # return jnp.linalg.norm(x).reshape(1)  # Condition on radius
  adjoint_observation_map = None
  H = None

  ddim_methods = ['PiGDMVP', 'PiGDMVE', 'DDIMVE', 'DDIMVP', 'KGDMVP', 'KGDMVE', 'STSL']
  cs_methods = ['KPDDPM', 'KPDDPMdiag', 'DPSDDPM', 'PiGDMVP', 'STSL']
  # cs_methods = ['TMPD2023avjp', 'TMPD2023bvjp', 'Chung2022scalar', 'Song2023']

  # Observation
  y = jnp.ones((config.eval.batch_size, 1)) * 0.7
  cs_method = cs_methods[0]
  config.sampling.cs_method = cs_method
  fn = model if cs_method in ddim_methods else true_score
  sampler = get_cs_sampler(
    config, sde, fn, (config.eval.batch_size//num_devices, config.data.image_size),
    None,  # dataset.get_data_inverse_scaler(config),
    y, H, observation_map, adjoint_observation_map,
    stack_samples=config.sampling.stack_samples)
  time_prev = time.time()
  samples, _ = sampler(sample_rng)
  sample_time = time.time() - time_prev
  print(samples.shape)
  plot_heatmap(samples=samples[:, [0, 1]], area_bounds=[-3., 3.], fname="diffusion_posterior_heatmap")
  assert 0

  # Sample true posterior using rayleigh
  # Compare using Wasserstein
  sliced_wasserstein_distance = sliced_wasserstein(dist_1=np.array(posterior_samples), dist_2=np.array(samples), n_slices=10000)
  ot_sliced_wasserstein_distance = ot_sliced_wasserstein(seed=seed_num_inv_problem, dist_1=np.array(posterior_samples), dist_2=np.array(samples), n_slices=10000)
  print("sample_time: {}, {}".format(sample_time, config.sampling.cs_method), sliced_wasserstein_distance, ot_sliced_wasserstein_distance)
  dists_infos.append({"seed": seed_num_inv_problem,
                      "dim": config.data.image_size,
                      "dim_y": dim_y,
                      "noise_std": config.sampling.noise_std,
                      "num_steps": config.solver.num_outer_steps,
                      "algorithm": config.sampling.cs_method,
                      "distance_name": 'sw',
                      "distance": sliced_wasserstein_distance,
                      "ot_distance": ot_sliced_wasserstein_distance
                      })
  plot_image(config.sampling.noise_std, dim, dim_y, 1000, i, cs_method, [0, 1], samples, posterior_samples)

  pd.DataFrame.from_records(dists_infos).to_csv(workdir + '/{}_{}_gmm_inverse_problem_comparison.csv'.format(config.sampling.cs_method, config.sampling.noise_std), float_format='%.3f')

  data = pd.read_csv(workdir + '/{}_{}_gmm_inverse_problem_comparison.csv'.format(config.sampling.cs_method, config.sampling.noise_std))
  agg_data = data.groupby(['dim', 'dim_y', 'num_steps', 'algorithm', 'distance_name']).distance.apply(lambda x: f'{np.nanmean(x):.1f} ± {1.96 * np.nanstd(x) / (x.shape[0]**.5):.1f}').reset_index()

  agg_data_sw = agg_data.loc[agg_data.distance_name == 'sw'].pivot(index=('dim', 'dim_y', 'num_steps'), columns='algorithm', values=['distance']).reset_index()
  agg_data_sw.columns = [col[-1].replace(' ', '_') if col[-1] else col[0].replace(' ', '_') for col in agg_data_sw.columns.values]

  for col in agg_data_sw.columns:
    if col not in ['dim', 'dim_y', 'num_steps']:
      agg_data_sw[col + '_num'] = agg_data_sw[col].apply(lambda x: float(x.split(" ± ")[0]))
  agg_data_sw.loc[agg_data_sw.num_steps == 1000].to_csv(workdir + '/{}_{}_gmm_inverse_problem_aggregated_sliced_wasserstein_1000_steps.csv'.format(config.sampling.cs_method, config.sampling.noise_std))


if __name__ == "__main__":
  app.run(main)
