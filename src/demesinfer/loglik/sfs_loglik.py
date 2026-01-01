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
    In tskit, given a tree sequence, to obtain the afs one can use the function
    ::
        afs = tree_sequence.allele_frequency_spectrum(*options)
    
    To obtain the esfs, with ``momi3`` one must first initialize an ExpectedSFS object
    with a ``demes`` demographic model and a dictionary of the number of samples used per population. 
    Then one would input a dictionary of parameter values into the Expected SFS object::
        ESFS_obj = demesinfer.sfs.ExpectedSFS(demes_model.to_demes(), num_samples=samples_per_population)
        params = {param_key: value}
        esfs = ESFS_obj(params)

        multinomial_loglik_value = sfs_loglik(afs, esfs)
        poisson_loglik_value = sfs_loglik(afs, esfs, sequence_length=1e8, theta=1e-8)

    To compute the gradient, one can use ``jax.grad`` or ``jax.value_and_grad``. 
    All loglikelihood functions are compatible with ``jax``.
    
    Please refer to the tutorial for a specfic example, the above provided codes are just outlines of how to call on the functions.
    
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
    """
    This function evaluates the **projected** multinomial or Poisson log-likelihood of an
    observed site frequency spectrum (AFS) given an expected spectrum (ESFS) via Einstein summation.

    By default, the sequence length and mutation rate (theta) are None, indicating
    that the multinomial likelihood will be used. To use the Poisson likelihood, one must
    provide BOTH the sequence length and mutation rate (theta).

    Parameters
    ----------
    esfs_obj : array_like
        An demesinfer.sfs.ExpectedSFS object
    sequence_length : int, optional
        Total number of sites in the sequence. Required if ``theta`` is given
    theta : float, optional
        Population-scaled mutation rate. If provided, a sequence length must also 
        be provided and the Poisson likelihood is used, 
        otherwise a multinomial likelihood is assumed. 
    proj_dict : dict 
        Dictionary of arrays that represent projection vectors
    einsum_str : string 
        Einstein summation string for projection
    input_arrays: array_like
        Input arrays for einsum operation, it must contain the original afs

    Returns
    -------
    float
        Log-likelihood of the projected observed spectrum given the projected expected spectrum.

    Notes
    -----
    proj_dict contains the random projection vectors that define the low-dimensional 
    subspace for approximating the full expected SFS, einsum_str is a string specifying 
    the Einstein summation for tensor operations, and input_arrays are preprocessed arrays 
    that serve as inputs to the jax.numpy.einsum call, optimized for JAX’s just-in-time compilation

    Example:
    ::
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
        esfs_obj = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)
        params = {param_key: val}
        projection_sfs_loglik(esfs_obj, params, proj_dict, einsum_str, input_arrays, sequence_length=None, theta=None)
    
    Internally this function will call on demesinfer.sfs.ExpectedSFS.tensor_prod, which performs the projection
    operations on the site frequency spectrum.

    Please refer to the tutorial for a specfic example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.sfs.ExpectedSFS
    demesinfer.sfs.ExpectedSFS.tensor_prod
    demesinfer.sfs.sfs_loglik.prepare_projection
    """
    proj_esfs = tp(proj_dict)
    proj_afs = jnp.einsum(einsum_str, *input_arrays)

    if theta:
        assert(sequence_length)
        tmp = proj_esfs * sequence_length * theta
        return jnp.sum(-tmp + xlogy(proj_afs, tmp))
    else:
        return jnp.sum(xlogy(proj_afs, proj_esfs/jnp.sum(proj_esfs)))
