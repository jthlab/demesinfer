import jax.numpy as jnp
from jax.scipy.special import xlogy

def sfs_loglik(afs, esfs, sequence_length=None, theta=None):
    """
    Compute the log-likelihood of an observed allele frequency spectrum.

    This function evaluates the multinomial or Poisson log-likelihood of an
    observed site frequency spectrum (AFS) given an expected spectrum (ESFS).

    Parameters
    ----------
    afs : array_like
        Observed allele frequency spectrum. The first and last entries
        (monomorphic classes) are ignored.
    esfs : array_like
        Expected allele frequency spectrum. Must be the same shape as ``afs``.
    sequence_length : int, optional
        Total number of sites in the sequence. Required if ``theta`` is given.
    theta : float, optional
        Population-scaled mutation rate. If provided, a Poisson likelihood
        is used; otherwise a multinomial likelihood is assumed.

    Returns
    -------
    loglik : float
        Log-likelihood of the observed spectrum given the expected spectrum.

    Notes
    -----
    If ``theta`` is provided, the likelihood is computed as::

        sum(-λ + afs * log(λ))

    where ``λ = esfs * sequence_length * theta``.

    Otherwise, the expected spectrum is normalized and a multinomial
    likelihood is computed.

    See Also
    --------
    demesinfer.fit.fit_model

    """
    afs = afs.flatten()[1:-1]
    esfs = esfs.flatten()[1:-1]
    
    if theta:
        assert(sequence_length)
        tmp = esfs * sequence_length * theta
        return jnp.sum(-tmp + xlogy(afs, tmp))
    else:
        return jnp.sum(xlogy(afs, esfs/esfs.sum()))

def projection_sfs_loglik(afs, tp, proj_dict, einsum_str, input_arrays, sequence_length=None, theta=None):
    proj_esfs = tp(proj_dict)
    proj_afs = jnp.einsum(einsum_str, *input_arrays)

    if theta:
        assert(sequence_length)
        tmp = proj_esfs * sequence_length * theta
        return jnp.sum(-tmp + xlogy(proj_afs, tmp))
    else:
        return jnp.sum(xlogy(proj_afs, proj_esfs/jnp.sum(proj_esfs)))
