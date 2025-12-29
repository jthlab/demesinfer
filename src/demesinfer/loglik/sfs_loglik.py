import jax.numpy as jnp
from jax.scipy.special import xlogy

def sfs_loglik(afs, esfs, sequence_length=None, theta=None):
    """
    This function evaluates the multinomial or Poisson log-likelihood of an
    observed site frequency spectrum (AFS) given an expected spectrum (ESFS).

    By default, the sequence length and mutation rate (theta) are None, indicating
    that the multinomial likelihood will be used. To use the Poisson likelihood, one must
    provide BOTH the sequence length and mutation rate (theta).

    Parameters
    ----------
    afs : array_like
        Observed allele frequency spectrum
    esfs : array_like
        Expected allele frequency spectrum. Must be the same shape as ``afs``
    sequence_length : int, optional
        Total number of sites in the sequence. Required if ``theta`` is given
    theta : float, optional
        Population-scaled mutation rate. If provided, a sequence length must also 
        be provided and the Poisson likelihood is used; 
        otherwise a multinomial likelihood is assumed. 

    Returns
    -------
    float
        Log-likelihood of the observed spectrum given the expected spectrum.

    Notes
    -----
    In tskit, given a tree sequence, to obtain the afs one can use the function::
        tree_sequence.allele_frequency_spectrum()
    
    To obtain the esfs, with ``momi3`` one must first initialize an ExpectedSFS object
    with a demographic model and a dictionary of the number of samples used per population. 
    Then one would input a dictionary of parameter values into the Expected SFS object::
        ESFS = demesinfer.sfs.ExpectedSFS(demes_model.to_demes(), num_samples=samples_per_population)
        params = {param_key: value}
        esfs = ESFS(params)

    Please refer to the tutorial for a specfic example.
    
    See Also
    --------
    demesinfer.sfs.ExpectedSFS
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
