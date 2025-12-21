Optimization
========
This page demonstrates the entire constrained optimization workflow with ``momi3``. It stands on its own, separate from the main tutorial.

This tutorial shows how to create a customizable SciPy optimizer for any demographic model using ``scipy.minimize``. It is *not* a primer on numerical optimization. SciPy's API/behaviour can change across
versions — **We are not responsible for any updates/errors made to scipy.minimize.**

Requirements
------------

- Python 3.12+
- ``jax`` (CPU is fine)
- ``msprime>=1.3``
- ``scipy>=1.11``
- ``demesinfer``

Imports
-------

.. code-block:: python

   import numpy as np
   import jax
   import jax.numpy as jnp
   import msprime as msp
   from scipy.optimize import Bounds, LinearConstraint, minimize

   from demesinfer.sfs import ExpectedSFS
   from demesinfer.constr import EventTree, constraints_for
   from demesinfer.loglik.sfs_loglik import sfs_loglik


Revisiting the IWM model
------------------------

.. code-block:: python

   demo = msp.Demography()
   demo.add_population(initial_size=5000, name="anc")
   demo.add_population(initial_size=5000, name="P0")
   demo.add_population(initial_size=5000, name="P1")
   demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
   demo.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")

   sample_size = 10 # we simulate 10 diploids
   samples = {"P0": sample_size, "P1": sample_size}
   ts = msp.sim_mutations(
       msp.sim_ancestry(
           samples=samples, demography=demo,
           recombination_rate=1e-8, sequence_length=1e8, random_seed=12
       ),
       rate=1e-8, random_seed=13
   )

   afs_samples = {"P0": sample_size * 2, "P1": sample_size * 2}
   afs = ts.allele_frequency_spectrum(
       sample_sets=[ts.samples([1]), ts.samples([2])],
       span_normalise=False,
       polarised=True
   )

Inspecting and Selecting Parameters to optimize
--------------------------------------

Now that you have everything set up from the simulation, inspect the parameters you can work with:

.. code-block:: python    

    et = EventTree(demo.to_demes())
    et.variables

The variables are shown below; some are grouped in a ``frozenset`` to indicate parameters that must be optimized together. If you're unfamiliar with the grouping we use here, see the Tutorial and Notation section.

.. code-block:: python

    [frozenset({('demes', 0, 'epochs', 0, 'end_size'),
            ('demes', 0, 'epochs', 0, 'start_size')}),
    frozenset({('demes', 1, 'epochs', 0, 'end_size'),
               ('demes', 1, 'epochs', 0, 'start_size')}),
    frozenset({('demes', 2, 'epochs', 0, 'end_size'),
               ('demes', 2, 'epochs', 0, 'start_size')}),
    frozenset({('demes', 1, 'proportions', 0)}),
    frozenset({('demes', 2, 'proportions', 0)}),
    frozenset({('migrations', 0, 'rate')}),
    frozenset({('migrations', 1, 'rate')}),
    frozenset({('demes', 0, 'epochs', 0, 'end_time'),
               ('demes', 1, 'start_time'),
               ('demes', 2, 'start_time'),
               ('migrations', 0, 'start_time'),
               ('migrations', 1, 'start_time')}),
    frozenset({('demes', 1, 'epochs', 0, 'end_time'),
               ('demes', 2, 'epochs', 0, 'end_time'),
               ('migrations', 0, 'end_time'),
               ('migrations', 1, 'end_time')})]

Suppose now we wish to optimize the following parameters, their associated values will be the initial guesses in the optimization process. We collect those parameters into a dictionary:

.. code-block:: python

   paths = {frozenset({('demes', 0, 'epochs', 0, 'end_size'),
            ('demes', 0, 'epochs', 0, 'start_size')}):3000.,
           frozenset({('demes', 1, 'epochs', 0, 'end_size'),
               ('demes', 1, 'epochs', 0, 'start_size')}): 6000.,
           frozenset({('demes', 2, 'epochs', 0, 'end_size'),
               ('demes', 2, 'epochs', 0, 'start_size')}): 4000.}

    cons = create_constraints(demo.to_demes(), paths)

For any contrained optimization method, one needs a set of parameters they wish to optimize, the constraints, a way to compute the expected SFS, and a way to compute the likelihood and its gradient.

Initial Required Setup
----------------------

    ###### Part 1 #####
    path_order: List[Var] = list(paths) # convert parameters into list
    x0 = jnp.array(_dict_to_vec(paths, path_order)) # convert initial values into a vector
    lb = jnp.array([0, 0, 0])
    ub = jnp.array([1e8, 1e8, 1e8])
    afs = jnp.array(afs)
    esfs = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)

    ###### Part 2 #####
    seed = 5
    projection = False
    sequence_length = None
    theta = None

    # This if statements creates the random projection vector
    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    ##### Part 3 ######
    args_nonstatic = (path_order, proj_dict, input_arrays, sequence_length, theta, projection, afs)
    args_static = (esfs, einsum_str)
    L, LinvT = make_whitening_from_hessian(_compute_sfs_likelihood, x0, args_nonstatic, args_static)
    preconditioner_nonstatic = (x0, LinvT)
    g = pullback_objective(_compute_sfs_likelihood, args_static)
    y0 = np.zeros_like(x0)

    lb_tr = L.T @ (lb - x0)
    ub_tr = L.T @ (ub - x0)

For ``scipy.minimize``, the setup requires three parts. 

Part 1: A ``dictionary`` object is convenient for tracking the current state of parameters, but most optimizers can only operate over vectors/lists. So one must convert the ``dictionary`` object into a set of vectors/lists. In our experience with ``scipy.minimize``, it works best when you provide a lower bound ``lb`` and upper bound ``up`` that limits the parameter space that the optimizer searches over. To compute the expected SFS for any arbitrary set of parameter values, one must create the ``ExpectedSFS`` object.

Part 2: One must make two decisions, the first decision is to use a boolean ``projection`` to indicate whether we will be using the random projection as an approximation of the expected SFS. Here ``projection = False`` indicates we do not use random projections and instead calculate the expected SFS exactly. The second decision is the type of likelihood to use, one must specify **BOTH** ``sequence_length`` and ``theta`` to use the Poisson likelihood, otherwise leave both as ``None`` to use the Multinomial likelihood. 

Part 3: This setup is optional and will depend on the user's preference. In our experience with ``scipy.minimize``, due to the magnitude difference between parameters such as the migration rate and population sizes, the gradient with respect to each variable will cause parameter updates to be unstable. We implement a preconditioning method that makes our problem more suitable for optimization. Instead of optimizing over the classical likelihood function ``_compute_sfs_likelihood``, we transform the likelihood function and the bounds to instead optimize over a function ``g`` with better conditioning. For more information on preconditioning please refer to: https://en.wikipedia.org/wiki/Preconditioner

Constraints and scipy.optimize.LinearConstraint
----------------------

Use ``constraints_for`` to derive the linear constraints for your chosen
parameters. It returns a dict with:

- ``"eq"`` → ``(Aeq, beq)`` for equality constraints
- ``"ineq"`` → ``(G, h)`` for inequalities.

These map directly to SciPy's ``~scipy.optimize.LinearConstraint``:

.. code-block:: python

   linear_constraints: list[LinearConstraint] = []
    
    Aeq, beq = cons["eq"]
    A_tilde = Aeq @ LinvT
    b_tilde = beq - Aeq@x0
    if Aeq.size:
        linear_constraints.append(LinearConstraint(A_tilde, b_tilde, b_tilde))

    G, h = cons["ineq"]
    if G.size:
        linear_constraints.append(create_inequalities(G, h, LinvT, x0, size=len(paths)))

As explained in the Tutorial, one would use ``create_inequalities`` to modify the output of ``constraints_for`` into the appropriate scipy.optimize.LinearConstraint format. 

Create and run the optimizer
-----------------

The final step is use an optimizer. Here we use ``scipy.minimize`` with constrained optimizer ``"trust-constr"``. Please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

.. code-block:: python
    gtol = 1e-5
    xtol = 1e-5
    maxiter = 1000
    barrier_tol = 1e-5

    res = minimize(
        fun=neg_loglik,
        x0=y0,
        jac=True,
        args = (g, preconditioner_nonstatic, args_nonstatic, lb_tr, ub_tr),
        method=method,
        constraints=linear_constraints,
        options={
            'gtol': gtol,
            'xtol': xtol, 
            'maxiter': maxiter,
            'barrier_tol': barrier_tol,
        }
    )

    x_opt = np.array(x0) + LinvT @ res.x

Due to preconditioning, we must transform the variable and to inspect our final estimates:

.. code-block:: python

   print("Optimal parameters: ", _vec_to_dict(jnp.asarray(res.x), path_order))
   print("\nFinal likelihood evaluation: ", res.fun)
   print("Optimal parameters as a vector: ", x_opt)

For the simulated example, the estimates are close to the true values:

.. code-block:: python

   Optimal parameters:  
{frozenset({('demes', 0, 'epochs', 0, 'end_size'), ('demes', 0, 'epochs', 0, 'start_size')}): 5016.814453125, 

frozenset({('demes', 1, 'epochs', 0, 'start_size'), ('demes', 1, 'epochs', 0, 'end_size')}): 5238.4287109375, 

frozenset({('demes', 2, 'epochs', 0, 'start_size'), ('demes', 2, 'epochs', 0, 'end_size')}): 5025.2666015625}
   
   Final likelihood evaluation:  430751.8125
   Optimal parameters as a vector:  [5016.8145 5238.4287 5025.2666]

We have this full pipeline wrapped in a single ``fit_sfs`` function for convenience. See the API reference for available options and implementation details. The convenience of momi3 is that each component of the optimization pipeline can be modified and operated on its own, but if one wants to use the ``fit_sfs`` function:

.. code-block:: python



Visualizing the log-likelihood Landscape
--------------------------------------

One of the way to visualize this optimization process is to plot the log-likelihood surface
with a contour plot.

The basic idea is simple, we pick two parameters to scan over a grid, 
and for each grid point, we evaluate the log-likelihood and compile them to be plotted into a contour plot.

.. code-block:: python

    p1_key = frozenset({('demes', 0, 'epochs', 0, 'end_size'),
            ('demes', 0, 'epochs', 0, 'start_size')})
    p2_key = frozenset({('demes', 1, 'epochs', 0, 'end_size'),
            ('demes', 1, 'epochs', 0, 'start_size')})

    base = dict(paths)
    p1_center = float(base[p1_key])
    p2_center = float(base[p2_key])

    N1, N2 = 10, 10
    span = 2.0
    p1_vals = jnp.logspace(jnp.log10(p1_center/span), jnp.log10(p1_center*span), N1)
    p2_vals = jnp.logspace(jnp.log10(p2_center/span), jnp.log10(p2_center*span), N2)

    Z = np.full((N2, N1), np.nan, dtype=float)

    for i in range(N2):       
        p2 = float(p2_vals[i])
        for j in range(N1):   
            p1 = float(p1_vals[j])
            g = dict(base)
            g[p1_key] = jnp.asarray(p1, dtype=jnp.float64)
            g[p2_key] = jnp.asarray(p2, dtype=jnp.float64)

            x_vec = jnp.asarray([g[k] for k in path_order], dtype=jnp.float64)
            
            e_full = esfs(g)
            ll = sfs_loglik(afs, e_full, sequence_length=None, theta=None)
            Z[i, j] = float(ll)

    Z_dLL = Z - np.nanmax(Z)

    X, Y = np.meshgrid(np.asarray(p1_vals), np.asarray(p2_vals), indexing="xy")
    plt.figure(figsize=(7, 5.5))
    cs = plt.contour(X, Y, Z_dLL, levels=[-10, -5, -2, -1, -0.5, -0.2, -0.1], colors='black')
    plt.clabel(cs, inline=True, fontsize=8)
    cf = plt.contourf(X, Y, Z_dLL, levels=40, cmap='viridis')
    plt.colorbar(cf, label='negative log-likelihood')

    plt.xscale('log'); plt.yscale('log')
    plt.xlim(float(p1_vals.min()), float(p1_vals.max()))
    plt.ylim(float(p2_vals.min()), float(p2_vals.max()))
    plt.xlabel(str(p1_key))
    plt.ylabel(str(p2_key))
    plt.title('SFS log-likelihood contours')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

.. image:: images/Contour.png
   :alt: Contour plot of log-likelihood surface
   :align: center
