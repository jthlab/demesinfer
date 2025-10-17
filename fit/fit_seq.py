from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import msprime as msp
from scipy.optimize import LinearConstraint, minimize
import jax.random as jr
from jax import vmap, lax 

from demesinfer.constr import EventTree, constraints_for
from jax.scipy.special import xlogy
from demesinfer.sfs import ExpectedSFS
import numpy as np
from scipy.optimize import Bounds
from demesinfer.loglik.sfs_loglik import sfs_loglik, projection_sfs_loglik

from loguru import logger
logger.disable("demesinfer")

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _dict_to_vec(d: Params, keys: Sequence[Var]) -> jnp.ndarray:
    return jnp.asarray([d[k] for k in keys], dtype=jnp.float64)

def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    return {k: v[i] for i, k in enumerate(keys)}

def _vec_to_dict(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, float]:
    return {k: float(v[i]) for i, k in enumerate(keys)}

def create_bounds(param_list, lower_bound=0.0, upper_bound=0.1):
    """
    Create bounds where any tuple parameter with 'migration' in first position is bounded
    """
    n_params = len(param_list)
    lb_list = [-np.inf] * n_params
    ub_list = [np.inf] * n_params
    
    for i, param in enumerate(param_list):
        if isinstance(param, tuple) and "migration" in str(param[0]):
            lb_list[i] = lower_bound
            ub_list[i] = upper_bound
    
    return Bounds(lb=lb_list, ub=ub_list)

def plot_sfs_likelihood(demo, paths, vec_values, afs, afs_samples, num_projections = 200, seed = 5, projection=False, theta=None, sequence_length=None):
    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None
    
    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        params = _vec_to_dict_jax(vec_array, path_order)
        
        if projection:
            tp = jax.jit(lambda X: esfs.tensor_prod(X, params))
            return -projection_sfs_loglik(afs, tp, proj_dict, einsum_str, input_arrays, sequence_length, theta)
        else:
            e1 = esfs(params)
            return -sfs_loglik(afs, e1, sequence_length, theta)

    # Outer vmap: Parallelize across vec_values
    # batched_neg_loglik = vmap(evaluate_at_vec)  # in_axes=0 is default

    # 3. Compute all values (runs on GPU/TPU if available)
    # results = batched_neg_loglik(vec_values) 
    results = lax.map(evaluate_at_vec, vec_values)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def prepare_projection(afs, afs_samples, sequence_length, num_projections, seed):
    rng = np.random.default_rng(seed)
    proj_dict = {}
    pop_names = list(afs_samples.keys())
    n_dims = afs.ndim
    
    for i in range(n_dims):
        if sequence_length is None:
            proj_dict[pop_names[i]] = rng.integers(0, 2, size=(num_projections, afs.shape[i]), dtype=jnp.int32)
        else:
            proj_dict[pop_names[i]] = rng.integers(0, 2, size=(num_projections, afs.shape[i]), dtype=jnp.int32)

    input_subscripts = ",".join([f"z{chr(97+i)}" for i in range(n_dims)])  # "za,zb,zc"
    tensor_subscript = "".join([chr(97+i) for i in range(n_dims)])         # "abc"
    output_subscript = "z"                                                 # "z"
    einsum_str = f"{input_subscripts},{tensor_subscript}->{output_subscript}"
    input_arrays = [proj_dict[pop_names[i]] for i in range(n_dims)] + [afs]

    return proj_dict, einsum_str, input_arrays

def plot_sfs_contour(demo, paths, param1_vals, param2_vals, afs, afs_samples, num_projections = 200, seed = 5, projection=False, theta=None, sequence_length=None):
    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None
    
    def compute_for_param1(param1_val):
        def compute_for_param2(param2_val):
            vec_array = jnp.array([param1_val, param2_val])
            params = _vec_to_dict_jax(vec_array, path_order)
    
            if projection:
                tp = jax.jit(lambda X: esfs.tensor_prod(X, params))
                return -projection_sfs_loglik(tp, proj_dict, einsum_str, input_arrays, sequence_length, theta)
            else:
                e1 = esfs(params)
                return -sfs_loglik(afs, e1, sequence_length, theta)
        
        # Map over param2 values for a fixed param1
        return jax.lax.map(compute_for_param2, param2_vals)
    
    # Map over param1 values
    log_likelihood_grid = jax.lax.map(compute_for_param1, param1_vals)
    
    param1_grid, param2_grid = jnp.meshgrid(param1_vals, param2_vals)
    param1_grid_np = np.array(param1_grid)
    param2_grid_np = np.array(param2_grid)
    log_likelihood_grid_np = np.array(log_likelihood_grid)
    
    plt.figure(figsize=(10, 8))
    
    # Use contourf for filled contours (heatmap instead of just lines)
    contour = plt.contourf(param1_grid_np, param2_grid_np, log_likelihood_grid_np.T, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Negative Log-Likelihood')
    
    contour_lines = plt.contour(param1_grid_np, param2_grid_np, log_likelihood_grid_np.T, levels=20, colors='black', linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Negative Log-Likelihood Contour Plot')
    plt.show()
    
    return param1_grid_np, param2_grid_np, log_likelihood_grid_np
    
def fit(
    demo,
    paths: Params,
    afs,
    afs_samples,
    *,
    method: str = "trust-constr",
    options: Optional[dict] = None,
    recombination_rate: float = None,
    sequence_length: float = None,
    theta: float = None,
    projection: bool = False,
    num_projections: float = 200,
    seed: float = 5, 
    gtol: float = 1e-4,
    xtol: float = 1e-4, #default 1e-8
    maxiter: int = 200, #default 1000
    barrier_tol: float = 1e-4,
    lower_bound: float = 0.0,
    upper_bound: float = 0.1,
    bounds = None,
):
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    et = EventTree(demo)
    
    if not bounds:
        bounds = create_bounds(paths)

    cons = constraints_for(et, *path_order)
    print(cons)
    print(bounds)
    linear_constraints: list[LinearConstraint] = []
    
    Aeq, beq = cons["eq"]
    if Aeq.size:
        linear_constraints.append(LinearConstraint(Aeq, beq, beq))

    G, h = cons["ineq"]
    if G.size:
        lower = -jnp.inf * jnp.ones_like(h)
        linear_constraints.append(LinearConstraint(G, lower, h))

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None
    
    @jax.value_and_grad
    def neg_loglik(vec):
        params = _vec_to_dict_jax(vec, path_order)

        if projection:
            tp = jax.jit(lambda X: esfs.tensor_prod(X, params))
            return -projection_sfs_loglik(afs, tp, proj_dict, einsum_str, input_arrays, sequence_length, theta)
        else:
            e1 = esfs(params)
            return -sfs_loglik(afs, e1, sequence_length, theta)

    res = minimize(
        fun=lambda x: float(neg_loglik(x)[0]),
        x0=jnp.asarray(x0),
        jac=lambda x: jnp.asarray(neg_loglik(x)[1], dtype=float),
        method=method,
        bounds = bounds,
        constraints=linear_constraints,
        options={
        'gtol': gtol,
        'xtol': xtol, #default 1e-8
        'maxiter': maxiter, #default 1000
        'barrier_tol': barrier_tol
        }
    )

    return _vec_to_dict(jnp.asarray(res.x), path_order), res
