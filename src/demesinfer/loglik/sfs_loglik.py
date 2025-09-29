import jax.numpy as jnp
from jax.scipy.special import xlogy

def sfs_loglik(afs, esfs, sequence_length=None, theta=None):
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
