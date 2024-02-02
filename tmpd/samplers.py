"""Samplers."""
from diffusionjax.utils import get_sampler, shared_update, get_times, get_linear_beta_function, get_sigma_function
from diffusionjax.inverse_problems import (
    get_dps,
    get_diffusion_posterior_sampling,
    get_pseudo_inverse_guidance,
    get_jacrev_guidance,
    get_jacfwd_guidance,
    get_vjp_guidance_mask,
    get_vjp_guidance,
    get_diag_jacrev_guidance,
    get_diag_vjp_guidance,
    get_diag_jacfwd_guidance)
from diffusionjax.solvers import EulerMaruyama
from tmpd.solvers import (
    STSL, MPGD,
    DPSDDPM, DPSDDPMplus,
    KPDDPM, KPSMLD,
    KPSMLDdiag, KPDDPMdiag,
    KPDDPMplus, KPSMLDplus,
    DPSSMLD, DPSSMLDplus,
    PiGDMVE, PiGDMVP, PiGDMVPplus, PiGDMVEplus, ReproducePiGDMVP, ReproducePiGDMVPplus,
    KGDMVE, KGDMVP, KGDMVEplus, KGDMVPplus)
from functools import partial
import jax.numpy as jnp
from jax import random
from jax.lax import scan


def get_cs_sampler(config, sde, model, sampling_shape, inverse_scaler, y, H, observation_map, adjoint_observation_map, stack_samples=False):
    """Create a sampling function

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A valid SDE class (the forward sde).
        score:
        shape: The shape of array, x. (num_samples,) + x_shape, where x_shape is the shape
            of the object being sampled from, for example, an image may have
            x_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
        inverse_scaler: The inverse data normalizer function.
        y: the data
        H: an observation matrix.
        operator_map:
        adjoint_operator_map: TODO generalize like this?

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """
    if config.sampling.cs_method.lower()=='chung2022scalar' or config.sampling.cs_method.lower()=='chung2022scalarplus':
        scale = config.solver.num_outer_steps * config.solver.dps_scale_hyperparameter
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_dps, observation_map, y, scale)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='chung2022' or config.sampling.cs_method.lower()=='chung2022plus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diffusion_posterior_sampling, observation_map, y, config.sampling.noise_std)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='song2023':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_pseudo_inverse_guidance, observation_map, y, config.sampling.noise_std, H @ H.T)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='song2023plus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_pseudo_inverse_guidance, observation_map, y, config.sampling.noise_std)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023avjp':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                get_vjp_guidance, H, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023ajacfwd':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_jacfwd_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023ajacrev' or config.sampling.cs_method.lower()=='tmpd2023ajacrevplus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_jacrev_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023b':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diag_jacrev_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023bjacfwd':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diag_jacfwd_guidance, observation_map, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023bvjp':  # This vmaps across calculating N_y vjps, so is O(num_samples * num_y * prod(shape)) in memory
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_diag_vjp_guidance, H, y, config.sampling.noise_std, sampling_shape)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='tmpd2023bvjpplus':
        sampler = get_sampler(sampling_shape,
                              EulerMaruyama(sde.reverse(model).guide(
                                  get_vjp_guidance_mask, observation_map, y, config.sampling.noise_std)),
                              inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples,
                              denoise=True)
    elif config.sampling.cs_method.lower()=='stsl':
        # Reproduce Second Order Tweedie sampler from Surrogate Loss (STSL) from (Rout et al. 2023)
        # What contrastive loss do they use?
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        forward_solver = EulerMaruyama(sde)
        outer_solver = STSL(config.solver.stsl_scale_hyperparameter,
                            config.solver.dps_scale_hyperparameter, y,
                            observation_map, adjoint_observation_map,
                            config.sampling.noise_std, sampling_shape[1:], model, config.solver.eta, beta, ts)
        # needs a special prior initialization that depends on the observed data, so uses different sampler
        sampler = get_stsl_sampler(sampling_shape,
                                   forward_solver,
                                   outer_solver,
                                   inverse_scaler=inverse_scaler,
                                   stack_samples=stack_samples,
                                   denoise=True)
    elif config.sampling.cs_method.lower()=='mpgd':
        # Reproduce MPGD (et al. 2023) paper for VP SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        outer_solver = MPGD(config.solver.mpgd_scale_hyperparameter, y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler,
                              stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpm':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        score = model
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        outer_solver = DPSDDPM(config.solver.dps_scale_hyperparameter, y, observation_map, score, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpsddpmplus':
        score = model
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        # Reproduce DPS (Chung et al. 2022) paper for VP SDE
        # https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L86
        # https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py#L2[…]C47
        outer_solver = DPSDDPMplus(config.solver.dps_scale_hyperparameter, y, observation_map, score, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmld':
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        score = model
        outer_solver = DPSSMLD(config.solver.dps_scale_hyperparameter, y, observation_map, score, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='dpssmldplus':
        # Reproduce DPS (Chung et al. 2022) paper for VE SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        score = model
        outer_solver = DPSSMLDplus(config.solver.dps_scale_hyperparameter, y, observation_map, score, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpm':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                           beta_max=config.model.beta_max)
        score = model
        outer_solver = KPDDPM(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmld':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        score = model
        outer_solver = KPSMLD(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmlddiag':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        score = model
        outer_solver = KPSMLDdiag(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpmdiag':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        score = model
        outer_solver = KPDDPMdiag(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpddpmplus':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        score = model
        outer_solver = KPDDPMplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kpsmldplus':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        score = model
        outer_solver = KPSMLDplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], score, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='reproducepigdmvp':
        # Reproduce PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        outer_solver = ReproducePiGDMVP(y, observation_map, config.sampling.noise_std, sampling_shape[:1], model, data_variance=1., eta=config.solver.eta, beta=beta, ts=ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='reproducepigdmvpplus':
        # Reproduce PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                    beta_max=config.model.beta_max)
        outer_solver = ReproducePiGDMVPplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, data_variance=1., eta=config.solver.eta, beta=beta, ts=ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmvp':
        # Based on PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                 beta_max=config.model.beta_max)
        outer_solver = PiGDMVP(y, observation_map, config.sampling.noise_std, sampling_shape[:1], model, data_variance=1., eta=config.solver.eta, beta=beta, ts=ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmvpplus':
        # Based on PiGDM (Song et al. 2023) paper for VP SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                    beta_max=config.model.beta_max)
        outer_solver = PiGDMVPplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, data_variance=1., eta=config.solver.eta, beta=beta, ts=ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmve':
        # Reproduce PiGDM (Song et al. 2023) paper for VE SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        outer_solver = PiGDMVE(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, data_variance=1., eta=config.solver.eta, sigma=sigma, ts=ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='pigdmveplus':
        # Reproduce PiGDM (Song et al. 2023) paper for VE SDE
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        outer_solver = PiGDMVEplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, data_variance=1., eta=config.solver.eta, sigma=sigma, ts=ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmvp':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                    beta_max=config.model.beta_max)
        outer_solver = KGDMVP(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmve':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        outer_solver = KGDMVE(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmvpplus':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        beta, _ = get_linear_beta_function(beta_min=config.model.beta_min,
                                    beta_max=config.model.beta_max)
        outer_solver = KGDMVPplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, beta, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='kgdmveplus':
        ts, _ = get_times(num_steps=config.solver.num_outer_steps,
                       dt=config.solver.dt, t0=config.solver.epsilon)
        sigma = get_sigma_function(
          sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max)
        outer_solver = KGDMVEplus(y, observation_map, config.sampling.noise_std, sampling_shape[1:], model, sigma, ts)
        sampler = get_sampler(sampling_shape, outer_solver, inverse_scaler=inverse_scaler, stack_samples=stack_samples, denoise=True)
    elif config.sampling.cs_method.lower()=='wc4dvar':
        # Get prior samples via a method
        raise NotImplementedError("TODO")
    else:
        raise ValueError("`config.sampling.cs_method` not recognized, got {}".format(config.sampling.cs_method))
    return sampler


def get_stsl_sampler(shape,
                     forward_solver,
                     inner_solver,
                     denoise=True,
                     stack_samples=False,
                     inverse_scaler=None):
  """Get a sampler from (possibly interleaved) numerical solver(s).

  Args:
    shape: Shape of array, x. (num_samples,) + obj_shape, where x_shape is the shape
      of the object being sampled from, for example, an image may have
      obj_shape==(H, W, C), and so shape==(N, H, W, C) where N is the number of samples.
    inner_solver: A valid numerical solver class that will act on an inner loop.
    denoise: Bool, that if `True` applies one-step denoising to final samples.
    stack_samples: Bool, that if `True` return all of the sample path or
      just returns the last sample.
    inverse_scaler: The inverse data normalizer function.
  Returns:
    A sampler.
  """
  if inverse_scaler is None: inverse_scaler = lambda x: x

  def sampler(rng, x_0=None):
    """
    Args:
      rng: A JAX random state.
      x_0: Initial condition. If `None`, then samples an initial condition from the
          sde's initial condition prior. Note that this initial condition represents
          `x_T sim Normal(O, I)` in reverse-time diffusion.
    Returns:
        Samples and the number of score function (model) evaluations.
    """
    forward_update = partial(shared_update,
                             solver=forward_solver)
    outer_ts = inner_solver.ts

    def forward_step(carry, t):
      rng, x, x_mean = carry
      vec_t = jnp.full((shape[0],), t)
      rng, step_rng = random.split(rng)
      x, x_mean = forward_update(step_rng, x, vec_t)
      return (rng, x, x_mean), ()

    inner_update = partial(shared_update,
                            solver=inner_solver)
    inner_ts = jnp.ones((3, 1))
    num_function_evaluations = jnp.size(outer_ts) * (3 * jnp.size(inner_ts))

    def inner_step(carry, t):
      rng, x, x_mean, vec_t = carry
      rng, step_rng = random.split(rng)
      x, x_mean = inner_update(step_rng, x, vec_t)
      return (rng, x, x_mean, vec_t), ()

    def outer_step(carry, t):
      rng, x, x_mean = carry
      vec_t = jnp.full(shape[0], t)
      rng, step_rng = random.split(rng)
      # Use x_mean as input for first inner step
      (rng, x, x_mean, vec_t), _ = scan(inner_step, (step_rng, x_mean, x_mean, vec_t), inner_ts)
      if not stack_samples:
        return (rng, x, x_mean), ()
      else:
        if denoise:
          return (rng, x, x_mean), x_mean
        else:
          return (rng, x, x_mean), x

    rng, step_rng = random.split(rng)
    if x_0 is None:
      x = inner_solver.prior(step_rng, shape)
    else:
      assert(x_0.shape==shape)
      x = x_0

    # get initial sample
    x = inner_solver.batch_adjoint_observation_map(inner_solver.y)
    (_, x, x_mean), _ = scan(forward_step, (rng, x, x), outer_ts, reverse=False)

    if not stack_samples:
      (_, x, x_mean), _ = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
      return inverse_scaler(x_mean if denoise else x), num_function_evaluations
    else:
      (_, _, _), xs = scan(outer_step, (rng, x, x), outer_ts, reverse=True)
      return inverse_scaler(xs), num_function_evaluations
  # return jax.pmap(sampler, in_axes=(0), axis_name='batch')
  return sampler
