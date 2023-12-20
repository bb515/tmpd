"""Animated GIF of sampling a bimodal with marginals colormapped image."""
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import jax
from jax.tree_util import Partial as partial
from jax import random
from jax import vmap, grad
import jax.numpy as jnp
from diffusionjax.run_lib import get_ddim_chain
from diffusionjax.utils import get_sampler
import diffusionjax.sde as sde_lib
from tmpd.samplers import get_cs_sampler
import torch
from torch import eye, randn_like, vstack, manual_seed
import numpyro.distributions as dist
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
import ot
from tqdm import trange
import time
from cycler import cycler
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

new_prop_cycle = cycler('color', ['r', 'g', 'b'])

BG_COLOUR = (239/255, 239/255, 239/255)
BG_ALPHA = 1.0
MG_ALPHA = 1.0
FG_ALPHA = 0.3
# color_posterior = '#a2c4c9'
# color_algorithm = '#ff7878'
dpi_val = 1200
cmap = 'magma'


ddim_methods = ['PiGDMVP', 'PiGDMVE', 'DDIMVE', 'DDIMVP', 'KGDMVP', 'KGDMVE']


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "./configs/animation.py", "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "./workdir", "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
logger = logging.getLogger(__name__)


def plot_animation(fig, ax, animate, frames, fname, fps=20, bitrate=800, dpi=300):
  ani = animation.FuncAnimation(
    fig, animate, frames=frames, interval=1, fargs=(ax,))
  # Set up formatting for the movie files
  Writer = animation.PillowWriter
  writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
  ani.save('{}.gif'.format(fname), writer=writer, dpi=dpi)


def plot_marginals1(x_lims, ou_dist, sde):
    res = 100
    hires = 2000
    t_lims = (1., 0.)
    x = np.linspace(x_lims[0], x_lims[1], res)
    x_hires = np.linspace(x_lims[0], x_lims[1], hires)
    # t_hires = np.array([0.80, 0.65, 0.45, 0.20])
    # t_hires = np.array([0.90, 0.55, 0.39, 0.10])
    # t_hires = np.array([0.95, 0.47, 0.35, 0.05])
    # t_hires = np.array([0.97, 0.46, 0.34, 0.03])
    t_hires = np.array([0.98, 0.45, 0.34, 0.01])
    # t_hires = np.array([1., 0.45, 0.33, 0.])
    t = np.linspace(t_lims[0], t_lims[1], res)
    xx, tt = np.meshgrid(x, t)
    xx_hires, tt_hires = np.meshgrid(x_hires, t_hires)
    xx = xx.flatten().reshape(-1, 1)
    tt = tt.flatten().reshape(-1, 1)
    xx_hires = xx_hires.flatten().reshape(-1, 1)
    tt_hires = tt_hires.flatten().reshape(-1, 1)
    fn = vmap(lambda x, t: ou_dist(sde.mean_coeff(t)).log_prob(x))
    vals = fn(xx, tt)
    hires_vals = fn(xx_hires, tt_hires)
    vals = vals.reshape((res, res))
    hires_vals = hires_vals.reshape((np.size(t_hires), hires))
    figure_sizes = {'fullwidth': 10}

    # Grid spec
    fig = plt.figure(figsize=(figure_sizes['fullwidth'], 0.4*figure_sizes['fullwidth']))
    fig.set_facecolor(BG_COLOUR)
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])
    gs.update(left=0.075,
          right=0.92,
          top=0.8,
          bottom=0.2,
          wspace=0.2,hspace=0.08)
    axs = np.array([fig.add_subplot(gs[0, i]) for i in range(4)])

    # ax = fig.add_subplot(gs[1, :])
    vmax = np.max(vals)
    vmin = np.min(vals)

    s = 8

    axs[0].scatter(jnp.exp(hires_vals[-1]), x_hires, c=hires_vals[-1], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)
    axs[1].scatter(jnp.exp(hires_vals[2]), x_hires, c=hires_vals[2], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)
    axs[2].scatter(jnp.exp(hires_vals[1]), x_hires, c=hires_vals[1], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)
    axs[3].scatter(jnp.exp(hires_vals[0]), x_hires, c=hires_vals[0], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)

    fontsize = 20
    # # axs[0].axis('off')
    # # axs[0].set_frame_on(False)
    # axs[0].set_xlabel("$p_{n=N}(x_{N})$", fontsize=fontsize)
    # # axs[0].get_xaxis().set_visible(False)
    # axs[0].get_yaxis().set_visible(False)

    # # axs[1].axis('off')
    # # axs[1].set_frame_on(False)
    # axs[1].set_xlabel("$p_{n=2}(x_{2})$", fontsize=fontsize)
    # # axs[1].get_xaxis().set_visible(False)
    # axs[1].get_yaxis().set_visible(False)

    # # axs[2].axis('off')
    # # axs[2].set_frame_on(False)
    # axs[2].set_xlabel("$p_{n=1}(x_{1})$", fontsize=fontsize)
    # # axs[2].get_xaxis().set_visible(False)
    # axs[2].get_yaxis().set_visible(False)

    # # axs[3].axis('off')
    # # axs[3].set_frame_on(False)
    # axs[3].set_xlabel("$p_{n=0}(x_{0})$", fontsize=fontsize)
    # # axs[3].get_xaxis().set_visible(False)
    # axs[3].get_yaxis().set_visible(False)


    axs[0].set_xlabel("$p_{n=0}(x_{0})$", fontsize=fontsize)
    # axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    # axs[1].axis('off')
    # axs[1].set_frame_on(False)
    axs[1].set_xlabel("$p_{n=1}(x_{1})$", fontsize=fontsize)
    # axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    # axs[2].axis('off')
    # axs[2].set_frame_on(False)
    axs[2].set_xlabel("$p_{n=2}(x_{2})$", fontsize=fontsize)
    # axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    # axs[3].axis('off')
    # axs[3].set_frame_on(False)
    axs[3].set_xlabel("$p_{n=N}(x_{N})$", fontsize=fontsize)
    # axs[3].get_xaxis().set_visible(False)
    axs[3].get_yaxis().set_visible(False)


    return fig, axs


def plot_marginals(x_lims, ou_dist, sde):
    res = 100
    hires = 2000
    t_lims = (1., 0.)
    x = np.linspace(x_lims[0], x_lims[1], res)
    x_hires = np.linspace(x_lims[0], x_lims[1], hires)
    t_hires = np.array([1., 0.])
    t = np.linspace(t_lims[0], t_lims[1], res)
    xx, tt = np.meshgrid(x, t)
    xx_hires, tt_hires = np.meshgrid(x_hires, t_hires)
    xx = xx.flatten().reshape(-1, 1)
    tt = tt.flatten().reshape(-1, 1)
    xx_hires = xx_hires.flatten().reshape(-1, 1)
    tt_hires = tt_hires.flatten().reshape(-1, 1)
    # fn = vmap(lambda x, t: jnp.exp(ou_dist(sde.mean_coeff(t)).log_prob(x)))
    fn = vmap(lambda x, t: ou_dist(sde.mean_coeff(t)).log_prob(x))
    vals = fn(xx, tt)
    hires_vals = fn(xx_hires, tt_hires)
    vals = vals.reshape((res, res))
    hires_vals = hires_vals.reshape((2, hires))

    figure_sizes = {'fullwidth': 10}
    # Grid spec
    fig = plt.figure(figsize=(figure_sizes['fullwidth'], 0.4*figure_sizes['fullwidth']))
    fig.set_facecolor(BG_COLOUR)

    gs = GridSpec(2, 3, width_ratios=[1, 3.3, 1], height_ratios=[1, 1])
    gs.update(left=0.075,
          right=0.92,
          top=0.8,
          bottom=0.2,
          wspace=0.2,hspace=0.08)
    axs = np.array([fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, :])])

    # ax = fig.add_subplot(gs[1, :])
    vmax = np.max(vals)
    vmin = np.min(vals)

    s = 8
    axs[0].scatter(jnp.exp(hires_vals[0]), x_hires, c=hires_vals[0], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)
    axs[1].imshow(vals.T, extent=(t_lims + x_lims), interpolation='bicubic', cmap='magma')
    axs[2].scatter(jnp.exp(hires_vals[-1]), x_hires, c=hires_vals[-1], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)

    # axs[0].axis('off')
    # axs[0].set_frame_on(False)
    axs[0].set_xlabel("$p_{t=1}(x_{t=1})$")
    # axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    axs[1].axis('off')
    axs[1].set_frame_on(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    # axs[2].axis('off')
    # axs[2].set_frame_on(False)
    axs[2].set_xlabel("$p_{t=0}(x_{t=0})$")
    # axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    bbox0 = axs[0].get_position()
    bbox1 = axs[1].get_position()
    bbox2 = axs[2].get_position()

    # im_ratio = 0.67
    im_ratio = 1.0
    width = 1.29
    axs[0].set_position([bbox1.x0 - 1.74 * width * bbox0.width, bbox0.y0 + bbox1.height * (1 - im_ratio) / 2, bbox0.width, bbox1.height * (im_ratio)])
    axs[1].set_aspect(0.33/(x_lims[1] - x_lims[0]))
    axs[2].set_position([bbox1.x0 + width * bbox0.width, bbox2.y0 + bbox1.height * (1 - im_ratio) / 2, bbox2.width, bbox1.height * (im_ratio)])

    bbox0 = axs[0].get_position()
    bbox1 = axs[1].get_position()

    # axs[3].axis('off')
    # axs[3].set_frame_on(False)
    axs[3].get_xaxis().set_visible(False)
    axs[3].get_yaxis().set_visible(False)
    axs[3].set_position([bbox1.x0, bbox1.y0 * (1. - 9./10), bbox1.height, bbox1.height])

    # fig.tight_layout()
    return fig, axs, (bbox0, bbox1)


def plot_posterior_marginals(x_lims, ou_dist, sde):
    res = 100
    hires = 2000
    t_lims = (1., 0.)
    x = np.linspace(x_lims[0], x_lims[1], res)
    x_hires = np.linspace(x_lims[0], x_lims[1], hires)
    t_hires = np.array([1., 0.])
    t = np.linspace(t_lims[0], t_lims[1], res)
    xx, tt = np.meshgrid(x, t)
    xx_hires, tt_hires = np.meshgrid(x_hires, t_hires)
    xx = xx.flatten().reshape(-1, 1)
    tt = tt.flatten().reshape(-1, 1)
    xx_hires = xx_hires.flatten().reshape(-1, 1)
    tt_hires = tt_hires.flatten().reshape(-1, 1)
    # fn = vmap(lambda x, t: jnp.exp(ou_dist(sde.mean_coeff(t)).log_prob(x)))
    fn = vmap(lambda x, t: ou_dist(sde.mean_coeff(t)).log_prob(x))
    vals = fn(xx, tt)
    hires_vals = fn(xx_hires, tt_hires)
    vals = vals.reshape((res, res))
    hires_vals = hires_vals.reshape((2, hires))

    figure_sizes = {'fullwidth': 10}
    # Grid spec
    fig = plt.figure(figsize=(figure_sizes['fullwidth'], 0.4*figure_sizes['fullwidth']))
    fig.set_facecolor(BG_COLOUR)
    gs = GridSpec(2, 3, width_ratios=[1, 3.3, 1], height_ratios=[1, 1])
    gs.update(left=0.075,
          right=0.92,
          top=0.8,
          bottom=0.2,
          wspace=0.2,hspace=0.08)
    axs = np.array([fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, :])])

    # ax = fig.add_subplot(gs[1, :])
    vmax = np.max(vals)
    vmin = np.min(vals)

    s = 8
    axs[0].scatter(jnp.exp(hires_vals[0]), x_hires, c=hires_vals[0], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)
    axs[1].imshow(vals.T, extent=(t_lims + x_lims), interpolation='bicubic', cmap='magma')
    axs[2].scatter(jnp.exp(hires_vals[-1]), x_hires, c=hires_vals[-1], edgecolor='None', cmap='magma', vmin=vmin, vmax=vmax, s=s)

    # axs[0].axis('off')
    # axs[0].set_frame_on(False)
    axs[0].set_xlabel("$p_{t=1}(x_{t=1})$")
    # axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    axs[1].axis('off')
    axs[1].set_frame_on(False)
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    # axs[2].axis('off')
    # axs[2].set_frame_on(False)
    axs[2].set_xlabel("$p_{t=0}(x_{t=0})$")
    # axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)

    bbox0 = axs[0].get_position()
    bbox1 = axs[1].get_position()
    bbox2 = axs[2].get_position()

    # im_ratio = 0.67
    im_ratio = 1.0
    width = 1.29
    axs[0].set_position([bbox1.x0 - 1.74 * width * bbox0.width, bbox0.y0 + bbox1.height * (1 - im_ratio) / 2, bbox0.width, bbox1.height * (im_ratio)])
    axs[1].set_aspect(0.33/(x_lims[1] - x_lims[0]))
    axs[2].set_position([bbox1.x0 + width * bbox0.width, bbox2.y0 + bbox1.height * (1 - im_ratio) / 2, bbox2.width, bbox1.height * (im_ratio)])

    bbox0 = axs[0].get_position()
    bbox1 = axs[1].get_position()

    # axs[3].axis('off')
    # axs[3].set_frame_on(False)
    axs[3].get_xaxis().set_visible(False)
    axs[3].get_yaxis().set_visible(False)
    axs[3].set_position([bbox1.x0, bbox1.y0 * (1. - 9./10), bbox1.height, bbox1.height])

    # fig.tight_layout()
    return fig, axs, (bbox0, bbox1)


def ot_sliced_wasserstein(seed, dist_1, dist_2, n_slices=100):
    return ot.sliced_wasserstein_distance(dist_1, dist_2, n_projections=n_slices, seed=seed)


def sliced_wasserstein(dist_1, dist_2, n_slices=100):
    # projections = torch.randn(size=(n_slices, dist_1.shape[1])).to(dist_1.device)
    projections = torch.randn(size=(n_slices, dist_1.shape[1]))
    projections = projections / torch.linalg.norm(projections, dim=-1)[:, None]
    dist_1_projected = (projections @ dist_1.T)
    dist_2_projected = (projections @ dist_2.T)
    return np.mean([wasserstein_distance(u_values=d1.cpu().numpy(), v_values=d2.cpu().numpy()) for d1, d2 in zip(dist_1_projected, dist_2_projected)])


def gaussian_posterior(y,
                       likelihood_A,
                       likelihood_bias,
                       likelihood_precision,
                       prior_loc,
                       prior_covar):
    prior_precision_matrix = torch.linalg.inv(prior_covar)
    posterior_precision_matrix = prior_precision_matrix + likelihood_A.T @ likelihood_precision @ likelihood_A
    posterior_covariance_matrix = torch.linalg.inv(posterior_precision_matrix)
    posterior_mean = posterior_covariance_matrix @ (likelihood_A.T @ likelihood_precision @ (y - likelihood_bias) + prior_precision_matrix @ prior_loc)
    try:
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix)
    except ValueError:
        u, s, v = torch.linalg.svd(posterior_covariance_matrix, full_matrices=False)
        s = s.clip(1e-12, 1e6).real
        posterior_covariance_matrix = u.real @ torch.diag_embed(s) @ v.real
        posterior_covariance_matrix = (posterior_covariance_matrix + posterior_covariance_matrix.T) / 2
        return MultivariateNormal(loc=posterior_mean, covariance_matrix=posterior_covariance_matrix)


def get_posterior(obs, prior, A, Sigma_y):
    modified_means = []
    modified_covars = []
    weights = []
    precision = torch.linalg.inv(Sigma_y)
    for loc, cov, weight in zip(prior.component_distribution.loc,
                                prior.component_distribution.covariance_matrix,
                                prior.mixture_distribution.probs):
        new_dist = gaussian_posterior(obs,
                                      A,
                                      torch.zeros_like(obs),
                                      precision,
                                      loc,
                                      cov)
        modified_means.append(new_dist.loc)
        modified_covars.append(new_dist.covariance_matrix)
        prior_x = MultivariateNormal(loc=loc, covariance_matrix=cov)
        residue = obs - A @ new_dist.loc
        log_constant = -(residue[None, :] @ precision @ residue[:, None]) / 2 + \
                       prior_x.log_prob(new_dist.loc) - \
                       new_dist.log_prob(new_dist.loc)
        weights.append(torch.log(weight).item() + log_constant)
    weights = torch.tensor(weights)
    weights = weights - torch.logsumexp(weights, dim=0)
    cat = Categorical(logits=weights)
    ou_norm = MultivariateNormal(loc=torch.stack(modified_means, dim=0),
                                 covariance_matrix=torch.stack(modified_covars, dim=0))
    return MixtureSameFamily(cat, ou_norm)


def build_extended_svd(A: torch.tensor):
    U, d, V = torch.linalg.svd(A, full_matrices=True)
    coordinate_mask = torch.ones_like(V[0])
    coordinate_mask[len(d):] = 0
    return U, d, coordinate_mask, V


def generate_measurement_equations(dim, dim_y, device, mixt, noise_std):
    A = torch.randn((dim_y, dim))

    u, diag, coordinate_mask, v = build_extended_svd(A)
    diag = torch.sort(torch.rand_like(diag), descending=True).values

    A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
    init_sample = mixt.sample()
    init_obs = A @ init_sample
    init_obs += randn_like(init_obs) * noise_std
    Sigma_y = torch.diag(noise_std**2 * torch.ones(len(diag)))
    posterior = get_posterior(init_obs, mixt, A, Sigma_y)
    return A, Sigma_y, u, diag, coordinate_mask, v, posterior, init_obs


def plot_single_image(noise_std, dim, dim_y, timesteps, i, name, indices, samples, color):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.hist(samples[:])
    # ax.scatter(*samples[:, indices].T, alpha=.5, color=color, edgecolors="black", rasterized=True)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlim([-22, 22])
    # ax.set_ylim([-22, 22])
    # fig.subplots_adjust(left=.005, right=.995,
    #                     bottom=.005, top=.995)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.pdf'.format(noise_std, dim, dim_y, timesteps, i, name), dpi=dpi_val)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.png'.format(noise_std, dim, dim_y, timesteps, i, name), transparent=True, dpi=dpi_val)
    plt.close(fig)


def plot_image(noise_std, dim, dim_y, timesteps, i, name, indices, diffusion_samples, target_samples=None):
    color_posterior = '#a2c4c9'
    color_algorithm = '#ff7878'
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.hist(target_samples[:], alpha=.5, color=color_posterior)
    ax.hist(diffusion_samples[:], alpha=.5, color=color_algorithm)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlim([-22, 22])
    # ax.set_ylim([-22, 22])
    # fig.subplots_adjust(left=.005, right=.995,
    #                     bottom=.005, top=.995)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.pdf'.format(noise_std, dim, dim_y, timesteps, i, name), dpi=dpi_val)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.png'.format(noise_std, dim, dim_y, timesteps, i, name), transparent=True, dpi=dpi_val)
    plt.close(fig)


def get_score_fn(ou_dist, sde):
    return vmap(grad(lambda x, t: ou_dist(sde.mean_coeff(t)).log_prob(x)))


def get_model_fn(ou_dist, sde):
    return vmap(grad(lambda x, t: - jnp.sqrt(sde.variance(t)) * ou_dist(sde.mean_coeff(t)).log_prob(x)))


def ou_mixt(mean_coeff, means, dim, weights):
    cat = Categorical(weights)
    ou_norm = MultivariateNormal(
        vstack(tuple((mean_coeff) * m for m in means)),
        eye(dim).repeat(len(means), 1, 1))
    return MixtureSameFamily(cat, ou_norm)


def ou_mixt_numpyro(mean_coeff, means, dim, weights):
    means = jnp.vstack(means) * mean_coeff
    covs = jnp.repeat(jnp.eye(dim)[None], axis=0, repeats=means.shape[0])
    return dist.MixtureSameFamily(component_distribution=dist.MultivariateNormal(
        means, covariance_matrix=covs), mixing_distribution=dist.Categorical(weights))


def main(argv):
    workdir = FLAGS.workdir
    config = FLAGS.config
    jax.default_device = jax.devices()[0]
    # Tip: use CUDA_VISIBLE_DEVICES to restrict the devices visible to jax
    # ... they must be all the same model of device for pmap to work
    num_devices =  int(jax.local_device_count()) if config.training.pmap else 1

    color_posterior = '#a2c4c9'
    color_algorithm = '#ff7878'

    # Torch device
    device = 'cpu'
    dists_infos = []

    # Setup SDE
    if config.training.sde.lower()=='vpsde':
        sde = sde_lib.VP(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    elif config.training.sde.lower()=='vesde':
        sde = sde_lib.VE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
    else:
        raise NotImplementedError(f"SDE {config.training.SDE} unknown.")

    ind_dim = 0
    ind_increase = 0
    size = 8.
    num_repeats = 1

    for ind_dim, dim in enumerate([1]):
        config.data.image_size = dim

        # setup of the inverse problem
        means = []
        for i in range(2):
            means += [torch.tensor([- size * i + size/2]).to(device)]
        weights = torch.ones(len(means))
        weights = weights / weights.sum()
        ou_mixt_fun = partial(ou_mixt,
                                means=means,
                                dim=config.data.image_size,
                                weights=weights)
        ou_mixt_jax_fun = partial(ou_mixt_numpyro,
                                    means=[jnp.array(m.numpy()) for m in means],
                                    dim=config.data.image_size,
                                    weights=jnp.array(weights.numpy()))

        rng = random.PRNGKey(config.seed)
        mixt_jax = ou_mixt_jax_fun(1)
        target_samples = mixt_jax.sample(rng, (config.eval.batch_size,))
        # plot_samples(target_samples, index=(0, 1), fname="target gmm jax")

        mixt = ou_mixt_fun(1)
        target_samples = mixt.sample((config.eval.batch_size,))
        logging.info("target prior:\nmean {},\nvar {}".format(np.mean(target_samples.numpy(), axis=0), np.var(target_samples.numpy(), axis=0)))

        # Plot prior samples
        score = get_score_fn(ou_mixt_jax_fun, sde)
        model = get_model_fn(ou_mixt_jax_fun, sde)

        # outer_solver = get_markov_chain(config, score)
        outer_solver = get_ddim_chain(config, model)
        inner_solver = None

        sampling_shape = (config.eval.batch_size//num_devices, config.data.image_size)
        sampler = get_sampler(sampling_shape, outer_solver,
                            inner_solver, denoise=config.sampling.denoise,
                            stack_samples=False)
        rng, sample_rng = random.split(rng, 2)
        samples, nfe = sampler(sample_rng)
        logging.info("diffusion prior:\nmean {},\nvar {}".format(np.mean(samples, axis=0), np.var(samples, axis=0)))
        plot_single_image(config.sampling.noise_std, dim, '_', 1000, i, 'prior', [0, 1], samples, color=color_algorithm)

        for ind_ptg, dim_y in enumerate([1]):
            for i in trange(0, num_repeats, unit="trials dim_y={}".format(dim_y)):
                seed_num_inv_problem = (2**(ind_dim))*(3**(ind_ptg)*(5**(ind_increase))) + i
                manual_seed(seed_num_inv_problem)

                A, Sigma_y, u, diag, coordinate_mask, v, posterior, init_obs = generate_measurement_equations(
                    config.data.image_size, dim_y, device, mixt, config.sampling.noise_std)
                # config.sampling.noise_std = float(Sigma_y.numpy()[0, 0])
                logging.info("ind_ptg {:d}, dim {:d}, dim_y {:d}, trial {:d}, noise_std {:.2e}".format(
                    ind_ptg, config.data.image_size, dim_y, i, config.sampling.noise_std))

                # Getting posterior samples form nuts
                posterior_samples_torch = posterior.sample((config.eval.batch_size,)).to(device)
                posterior_samples = posterior_samples_torch.numpy()
                plot_single_image(config.sampling.noise_std, dim, dim_y, 1000, i, 'posterior', [0, 1], posterior_samples, color=color_posterior)

                y = jnp.array(init_obs.numpy(), dtype=jnp.float32)
                y = jnp.tile(y, (config.eval.batch_size//num_devices, 1))
                H = jnp.array(A.numpy(), dtype=jnp.float32)

                def observation_map(x):
                    x = x.flatten()
                    return H @ x

                def adjoint_observation_map(y):
                    y = y.flatten()
                    return H.T @ y

                # cs_methods = ['KPDDPM', 'KPDDPMdiag', 'DPSDDPM', 'PiGDMVP']
                # cs_methods = ['TMPD2023avjp', 'TMPD2023bvjp', 'Chung2022scalar', 'Song2023']
                cs_methods = ['KPDDPM']
                # cs_methods = ['TMPD2023avjp']

                for cs_method in cs_methods:
                    config.sampling.cs_method = cs_method
                    fn = model if cs_method in ddim_methods else score
                    sampler = get_cs_sampler(
                        config, sde, fn, (config.eval.batch_size//num_devices, config.data.image_size),
                        None,  # dataset.get_data_inverse_scaler(config),
                        y, H, observation_map, adjoint_observation_map,
                        stack_samples=config.sampling.stack_samples)

                    time_prev = time.time()
                    samples, _ = sampler(sample_rng)
                    sample_time = time.time() - time_prev

                    if config.sampling.stack_samples:
                        ts = np.linspace(0., 1., config.solver.num_outer_steps + 1)[1:]
                        samples = samples.reshape(
                            config.solver.num_outer_steps, config.eval.batch_size, config.data.image_size)

                        frames = 100
                        x_lims = (-8., 8.)
                        fig, axs = plot_marginals1(x_lims, ou_mixt_jax_fun, sde)
                        fig.savefig("test0.png")
                        assert 0

                        fig, axs, (bbox0, bbox1) = plot_marginals(x_lims, ou_mixt_jax_fun, sde)
                        fig.savefig("test1.png")

                        axs[1].set_prop_cycle(new_prop_cycle)
                        axs[1].plot(ts[990].reshape(-1, 1), samples[990, 0, 0].reshape(-1, 1).T)
                        npz = np.load('./assets/0.05_FFHQ_KPSMLDplus_1.npz')
                        im_samples = npz['samples']
                        (num_outer_steps, _, image_size, _, num_channels) = im_samples.shape
                        bbox3_y0 = bbox1.y0 * (1. - 9./10)
                        def animate(i, axs):
                            for art in list(axs[1].lines):
                                art.remove()
                            idx = config.solver.num_outer_steps - int((i + 1) * config.solver.num_outer_steps / frames)
                            im_idx = num_outer_steps - int((i+1) * num_outer_steps / frames)
                            idxs = np.linspace(idx, int(config.solver.num_outer_steps * ( 1. - 1. / frames)), i + 1, dtype=int)
                            axs[1].plot(ts[idxs], samples[idxs, :, 0], linewidth="0.5")
                            bbox3_x0 = bbox0.x0 + bbox1.width * (i / frames)
                            axs[3].set_position([bbox3_x0, bbox3_y0, bbox1.height, bbox1.height])
                            im = im_samples[im_idx]
                            im = im.reshape((image_size, image_size, num_channels))
                            axs[3].imshow(im, interpolation=None)
                            print(samples[idxs[0], 0, 0])
                            sample_x0 = 256 * (0.5 + 3. / frames)
                            axs[1].annotate("",
                                        xy=(sample_x0, -150), xycoords=axs[3].transData,
                                        xytext=(sample_x0, 0), textcoords=axs[3].transData,
                                        arrowprops=dict(arrowstyle="<->"))
                            fig.savefig("{}-{}.png".format('kpsmldplus', i))

                        plot_animation(fig, axs,
                                    animate, frames,
                                    fname="{}_{}".format(0.05, 'kpsmldplus'),
                                    bitrate=600,
                                    dpi=300)

                    else:
                        samples = samples.reshape(config.eval.batch_size, config.data.image_size)
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



if __name__ == "__main__":
    app.run(main)
