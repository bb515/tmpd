"""Inverse problems. TODO: change so all with autodiff and forward maps."""
import jax.numpy as jnp
from jax import vmap, grad, jacfwd, vjp, jacrev, jvp, jit


def get_estimate_h_x_0(sde, score, shape, observation_map):
    """Get an MMSE estimate of x_0, pushed through observation_map.
    Args:
        observation_map: forward, assumed linear, map.
    """
    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expetion as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return observation_map((x + v_t * s) / m_t), s

    return estimate_x_0


def get_ratio(sde):
    if type(sde).__name__=='VE':
        def ratio(t):
            return sde.variance(t)
    elif type(sde).__name__=='VP':
        def ratio(t):
            return sde.variance(t) / sde.mean_coeff(t)
    else:
        raise ValueError("Did not recognise forward SDE (got {}, expected VE or VP)".format(type(sde).__name__))
    return ratio


def get_model_variance(sde):
    if type(sde).__name__=='VE':
        def model_variance(t):
            # return sde.variance(t) / (1 + sde.variance(t))  # This was unstable for Song et al. 2023
            return sde.variance(t)  # used this instead, as it is stable and a free hyperparameter in Song's method
    elif type(sde).__name__=='VP':
        def model_variance(t):
            return sde.variance(t)
    else:
        raise ValueError("Did not recognise SDE (got {}, expected VE or VP)".format(type(sde).__name__))
    return model_variance


def get_estimate_x_0(sde, score, shape):
    """Get an MMSE estimate of x_0
    """
    # TODO: problem is that score is already vmapped, does this cause any inefficiencies?
    # TODO: this can be a method within rsde class? no because it needs to reshape. So it could be
    # a method within the sampler class. This is probably the cleanest solution for now
    # since if it was in the sampler class, then sampler needs to access sde and is not general to
    # markov chains.
    def estimate_x_0(x, t):
        """The MMSE estimate for x_0|x_t,
        which is it's expectation as given by Tweedie's formula."""
        m_t = sde.mean_coeff(t)
        v_t = sde.variance(t)
        x = x.reshape(shape)
        x = jnp.expand_dims(x, axis=0)
        t = jnp.expand_dims(t, axis=0)
        s = score(x, t)
        s = s.flatten()
        x = x.flatten()
        return (x + v_t * s) / m_t, s

    return estimate_x_0


def get_dps(
        scale, sde, score, shape, y, noise_std, observation_map):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a vmap grad.
    """
    def get_l2_norm(y, estimate_h_x_0):
        def likelihood_score_approx(x, t):
            h_x_0, s = estimate_h_x_0(x, t)
            innovation = y - h_x_0
            return jnp.linalg.norm(innovation), s
        return likelihood_score_approx

    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape)
    l2_norm = get_l2_norm(y, estimate_h_x_0)
    likelihood_score = grad(l2_norm, has_aux=True)
    def approx_posterior_score(x, t):
        x = x.flatten()
        ls, s = likelihood_score(x, t)
        posterior_score = s - scale * ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_dps_plus(
        scale, sde, score, shape, y, noise_std, observation_map):
    """
    TODO: Unstable(?) and doesn't work on FFHQ_noise_std=0.001
    TODO: Unstable and doesn't work - weird oversmoothing of images
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a vmap grad.
    """
    def get_l2_norm(y, estimate_h_x_0):
        def l2_norm(x, t):
            # y is given as a d_x length vector
            h_x_0, s = estimate_h_x_0(x, t)
            innovation = y - h_x_0
            return jnp.linalg.norm(innovation), s
        return l2_norm

    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    l2_norm = get_l2_norm(y, estimate_h_x_0)
    likelihood_score = grad(l2_norm, has_aux=True)
    def approx_posterior_score(x, t):
        x = x.flatten()
        ls, s = likelihood_score(x, t)
        posterior_score = s - scale * ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diffusion_posterior_sampling_plus(
        sde, score, shape, y, noise_std, observation_map):
    """
    TODO: Unstable on FFHQ_noise_std=0.001
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a single vjp.
    NOTE: This is not how Chung et al. 2022, https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/guided_diffusion/condition_methods.py
    implemented their method, their method is get_dps_plus.
    Whereas this method uses their approximation in Eq. 11 https://arxiv.org/pdf/2209.14687.pdf#page=20&zoom=100,144,757
    to directly calculate the score.
    """
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        variance_estimate = jnp.linalg.norm(innovation)**2
        # C_yy = noise_std**2
        C_yy = variance_estimate  # TODO: remove this
        ls = innovation / C_yy
        ls = vjp_estimate_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diffusion_posterior_sampling(
        sde, score, shape, y, noise_std, observation_map):
    """
    `Diffusion Posterior Sampling for general noisy inverse problems'
    implemented with a single vjp.
    Assumes linear observation_map
    TODO: Generalizable to non-linear observation_map?
    """
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = noise_std**2
        f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_pseudo_inverse_guidance(
        sde, score, shape, y, h, h_dagger):
    """
    Pseudo-inverse guidance score for non-linear forward map h, with pseudo-inverse h_dagger.
    :arg shape:
    :arg y:
    :arg h:
    :arg h_dagger:
    :arg mask:
    :arg likelihood_score:
    :arg estimate_x_0:
    """
    def ratio(t):
        return sde.variance(t) / sde.mean_coeff(t)

    estimate_x_0 = get_estimate_x_0(sde, score, shape)
    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_estimate_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        innovation = h_dagger(y) - h_dagger(h(x_0))
        ls = vjp_estimate_x_0(innovation)[0]
        posterior_score = s + ls / ratio(t)
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance_plus(
        sde, score, shape, y, noise_std, observation_map):
    """
    Pseudo-Inverse guidance score for an observation_map that can be
    represented by a lambda x: mask * x
    """
    model_variance = get_model_variance(sde)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = model_variance(t) + noise_std**2
        f = innovation / C_yy
        ls = vjp_estimate_h_x_0(f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_linear_inverse_guidance(
        sde, score, shape, y, noise_std, observation_map, HHT):
    """Pseudo-Inverse guidance score for linear forward map observation_map.
    Args:
        HHT: H @ H.T which has shape (d_y, d_y)
    """
    model_variance = get_model_variance(sde)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        innovation = y - h_x_0
        C_yy = model_variance(t) * HHT + noise_std**2 * jnp.eye(y.shape[0])
        f = jnp.linalg.solve(C_yy, innovation)
        ls = vjp_estimate_h_x_0(f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diag_approximate_posterior(sde, score, shape, y, noise_std, observation_map):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    ratio = get_ratio(sde)
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = ratio(t) * jnp.diag(H_grad_H_x_0) + noise_std**2
        innovation = y - h_x_0
        f = innovation / C_yy
        ls = grad_H_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diag_vjp_approximate_posterior(sde, score, shape, y, noise_std, H):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    ratio = get_ratio(sde)
    observation_map = lambda x: H @ x
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_estimate_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        diag_vjp = vmap(lambda h: jnp.dot(vjp_estimate_h_x_0(h)[0], h))
        diag_H_grad_H_x_0 = diag_vjp(H.T)
        print(diag_H_grad_H_x_0.shape)
        assert 0
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = ratio(t) * diag_H_grad_H_x_0 + noise_std**2
        innovation = y - h_x_0
        f = innovation / C_yy
        ls = grad_H_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_diag_jacfwd_approximate_posterior(sde, score, shape, y, noise_std, observation_map):
    """Use a diagonal approximation to the variance inside the likelihood,
    This produces similar results when the covariance is approximately diagonal
    """
    ratio = get_ratio(sde)
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        H_grad_x_0 = jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = ratio(t) * jnp.diag(H_grad_H_x_0) + noise_std**2
        f = (y - h_x_0) / C_yy
        ls = H_grad_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score



def get_jvp_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible. TODO: profile vs jacfwd.
    """
    ratio = get_ratio(sde)
    observation_map = lambda x: H @ x
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        H_grad_x_0 = jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = ratio(t) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = H_grad_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_vjp_approximate_posterior(
        sde, score, shape, y, noise_std, H):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible. TODO: profile vs jvp methods.
    """
    ratio = get_ratio(sde)
    estimate_x_0 = get_estimate_h_x_0(sde, score, shape, lambda x: x)
    def approx_posterior_score(x, t):
        x = x.flatten()
        x_0, vjp_x_0, s = vjp(
            lambda x: estimate_x_0(x, t), x, has_aux=True)
        vec_vjp_x_0 = vmap(vjp_x_0)
        H_grad_x_0 = vec_vjp_x_0(H)[0]
        C_yy = ratio(t) * H @ H_grad_x_0.T + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - H @ x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = vjp_x_0(H.T @ f)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_vjp_approximate_posterior_plus(
        sde, score, shape, y, noise_std, observation_map):
    """
    TODO: This is incorrect... need to calculate full Jacobian and then take diagonal. However, it may perform well in practice?
    TODO: Unstable on FFHQ_noise_std=0.001
    Uses diagonal of second moment approximation of the covariance of x_0|x_t.

    Computes only two vjps.
    """
    ratio = get_ratio(sde)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, vjp_h_x_0, s = vjp(
            lambda x: estimate_h_x_0(x, t), x, has_aux=True)
        diag = observation_map(vjp_h_x_0(observation_map(jnp.ones(x.shape[0])))[0])
        C_yy = ratio(t) * diag + noise_std**2
        innovation = y - h_x_0
        ls = innovation / C_yy
        ls = vjp_h_x_0(ls)[0]
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jacrev_approximate_posterior(
        sde, score, shape, y, noise_std, observation_map):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using vjps where possible. TODO: profile vs jacfwd.
    """
    ratio = get_ratio(sde)
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        grad_H_x_0 = jacrev(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(grad_H_x_0)
        C_yy = ratio(t) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = grad_H_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score


def get_jacfwd_approximate_posterior(
        sde, score, shape, y, noise_std, observation_map):
    """
    Uses full second moment approximation of the covariance of x_0|x_t.

    Computes using d_y jvps.
    """
    ratio = get_ratio(sde)
    batch_observation_map = vmap(observation_map)
    estimate_h_x_0 = get_estimate_h_x_0(sde, score, shape, observation_map)
    def approx_posterior_score(x, t):
        x = x.flatten()
        h_x_0, s = estimate_h_x_0(x, t)  # TODO: in python 3.8 this line can be removed by utilizing has_aux=True
        H_grad_x_0 = jacfwd(lambda _x: estimate_h_x_0(_x, t)[0])(x)
        H_grad_H_x_0 = batch_observation_map(H_grad_x_0)
        C_yy = ratio(t) * H_grad_H_x_0 + noise_std**2 * jnp.eye(y.shape[0])
        innovation = y - h_x_0
        f = jnp.linalg.solve(C_yy, innovation)
        ls = H_grad_x_0.T @ f
        posterior_score = s + ls
        return posterior_score.reshape(shape)

    return approx_posterior_score

