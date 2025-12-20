Tutorial
========

This tutorial demonstrates how to use the ``momi3`` module (part of the ``demesinfer`` package) to perform demographic inference on population models. In momi3, the main approach for demographic inference is based on the site frequency spectrum (SFS) of genetic data.

The corresponding Jupyter notebook for this tutorial is available at ``docs/momi3_tutorial.ipynb``.

We will walk through simulating a population structure, running inference using the SFS, and interpreting the results.  

Simulation
----------

We begin by simulating genetic data under a simple demographic model using the ``msprime`` and ``demes`` packages.

To get started, import the necessary packages:

.. code-block:: python

    import msprime as msp
    import demes
    import demesdraw

For simplicity, we consider a classic isolation-with-migration (IWM) scenario: two subpopulations (P0 and P1) that split from a common ancestor. All populations are assumed to have constant effective population sizes of 5000, and after the split the subpopulations exchange migrants at a symmetric rate of 0.0001. We set the split time between the subpopulations and their ancestor to 1000 generations:

.. code-block:: python

    demo = msp.Demography()
    demo.add_population(initial_size=5000, name="anc")
    demo.add_population(initial_size=5000, name="P0")
    demo.add_population(initial_size=5000, name="P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    demo.add_population_split(time=1000, derived=[f"P{i}" for i in range(2)], ancestral="anc")

We can visualize the demographic model using ``demesdraw``:

.. code-block:: python

    g = demo.to_demes()
    demesdraw.tubes(g)

.. image:: images/demo.png
   :alt: Demographic model visualization
   :align: center

Next, we simulate the ancestry of 10 diploid individuals sampled from the two subpopulations using ``msprime.sim_ancestry()``.  
We use a mutation and recombination rate of 1e-8 and a sequence length of 100 million base pairs, with fixed random seeds for reproducibility.

.. code-block:: python

    sample_size = 10
    samples = {f"P{i}": sample_size for i in range(2)}
    anc = msp.sim_ancestry(samples=samples, demography=demo,
                           recombination_rate=1e-8,
                           sequence_length=1e8,
                           random_seed=12)
    ts = msp.sim_mutations(anc, rate=1e-8, random_seed=13)

Lastly, we compute the allele frequency spectrum (AFS) from the simulated data:

.. code-block:: python

    afs_samples = {f"P{i}": sample_size*2 for i in range(2)}
    afs = ts.allele_frequency_spectrum(
        sample_sets=[ts.samples([1]), ts.samples([2])],
        span_normalise=False,
        polarised=True
    )

For more details regarding the construction of demographic models using msprime.Demography(), please refer to: https://tskit.dev/msprime/docs/stable/demography.html

Demographic parameters in momi3
-------------------------------

A convenient feature of momi3 is its treatment of demographic model parameterization. It automatically translates a given demographic model (e.g., IWM, exponential growth, stepping stone) into the precise set of numerical constraints that satisfy model restrictions, such as those governing time intervals, population sizes, and admixture events. This eliminates the tedious and challenging manual derivation of constraints, making constrained optimization more accessible.

In the previous section, we simulated genetic data under an IWM model. We can now examine the full set of parameters associated with this model:

.. code-block:: python

    from demesinfer.constr import EventTree
    et = EventTree(g)
    et.variables

The output is a list of parameters, each entry representing an optimizable parameters in the IWM model. All of the parameters for ''demestats'' are contained in ''frozenset'' objects, and if variables are constrained to be equal by **construction** of the model, they appear grouped inside a frozenset:

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

Demes are indexed in the order they were added to the msprime.Demography() object. Here, demes 0, 1, and 2 correspond to populations anc, P0, and P1, respectively.

To gain a thorough understanding of the parameterization, please refer to the ''Notation Section''. In this specific example, any parameters within the same ``frozenset`` object are treated as a single parameter, which implicitly constrains them all to be equal. The first three frozenset objects represent the constant population sizes for anc, P0, and P1, respectively. Because the population size is constant over the epoch, the start and end size are treated as a single parameter. 

``frozenset({('migrations', 0, 'rate')})`` and ``frozenset({('migrations', 1, 'rate')})`` are the respective assymetric migration parameters between populations P0 and P1. By default they will be treated as assymetric, one can edit the constraints to enforce symmetry and constrain the optimization to treat the two directions of migration as a single parameter. (See section below on editing constraints)

Proportion parameters like ``frozenset({('demes', 1, 'proportions', 0)})`` and ``frozenset({('demes', 2, 'proportions', 0)})`` describe admixture or pulse events when a population is formed from multiple ancestors. In this simple IWM model, there are no admixture events, so they are trivial in this context (effectively fixed and unused), but still appear for consistency with the general framework.

The last two frozenset objects constrain the timing of events. Following the construction of the model, the start times of subpopulation and migration events must always match the end time of the ancestral population. The last ``frozenset`` constrains the end time of subpopulations and the end time migrations to align together.

Demographic constraints in momi3
-------------------------------
Suppose you were interested in inferring 3 parameters - the ancestral population size, rate of migration from P0 to P1, and the time of divergence. To output the associated linear constraints, we must input a list of the parameter into ''constraints_for''. 

.. code-block:: python

    from demesinfer.constr import constraints_for
    momi3_parameters = [
            frozenset({('demes', 0, 'epochs', 0, 'end_size'),
            ('demes', 0, 'epochs', 0, 'start_size')}), # Ancestral population size
            frozenset({('migrations', 0, 'rate')}), # Rate of migration
            frozenset({('demes', 0, 'epochs', 0, 'end_time'),
            ('demes', 1, 'start_time'),
            ('demes', 2, 'start_time'),
            ('migrations', 0, 'start_time'),
            ('migrations', 1, 'start_time')}) # Time of divergence
            ]
    constraint = constraints_for(et, *demestats_parameters)
    print(constraint)

If one does not want to use frozenset parameters, one can also optionally use the ''variable_for'' function to take ''demes'' paths and find the associated frozenset parameter. 

.. code-block:: python

    parameters = [
        ('demes', 0, 'epochs', 0, 'end_size'), # The ancestral population size
        ('migrations', 0, 'rate'), # Rate of migration from P0 to P1
        ('demes', 0, 'epochs', 0, 'end_time') # Time of divergence
    ]

    momi3_parameters = [et.variable_for(param) for param in parameters]
    constraint = constraints_for(et, *momi3_parameters)
    print(constraint)

The output of ``constraints_for`` is a dictionary with two keys:

.. code-block:: python

    {
    'eq': (array([], shape=(0, 3), dtype=float64),
            array([], dtype=float64)),

    'ineq': (array([[-1., -0., -0.],
                    [-0., -1., -0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  0., -1.]]),
            array([0., 0., 1., 0.]))
    }

``"eq"``: linear equality constraints will be a tuple of the form ``(A_eq, b_eq)`` such that ``A_eq @ x = b_eq``.

``"ineq"``: linear inequality constraints will be a tuple of the form ``(A_ineq, b_ineq)`` such that ``A_ineq @ x <= b_ineq``.

''constraints_for'' will preserve the ordering of the parameters, so we have:

- Column 0: ancestral population size
- Column 1: migration rate from P0 → P1
- Column 2: time of divergence

**Interpretation of inequality constraints**:

- Row 0: ``[-1., -0., -0.] <= 0`` → -(ancestral population size) <= 0 → ancestral population size ≥ 0
- Row 1: ``[-0., -1., -0.] <= 0`` → migration rate ≥ 0
- Row 2: ``[ 0.,  1.,  0.] <= 1`` → migration rate ≤ 1
- Row 3: ``[ 0.,  0., -1.] <= 0`` → split time ≥ 0

These constraints ensure meaningful parameter ranges: population sizes and times must be nonnegative, and migration rates must lie within the range of ``[0, 1]``.

In general, ``constraints_for`` automatically generates the linear constraints required for optimization. To verify and interpret the constraints more easily, one can optionally use the ''print_constraints'' function.

.. code-block:: python

    from demesinfer.constr import print_constraints
    print_constraints(constraint, momi3_parameters)

The output:

.. code-block:: python

    ==================================================
    Linear Equalities: Ax = b
    ==================================================
    
    None
    
    ==================================================
    Linear Inequalities: Ax <= b
    ==================================================
    
    CONSTRAINTS:
    --------------------------------------------------
    Row  1: -x1 ≤ 0
    Row  2: -x2 ≤ 0
    Row  3: x2 ≤ 1
    Row  4: -x3 ≤ 0
    --------------------------------------------------
    
    AS STRINGS:
    --------------------------------------------------
    Row 1: -frozenset({('demes', 0, 'epochs', 0, 'end_size'), ('demes', 0, 'epochs', 0, 'start_size')}) <= 0.0
    Row 2: -frozenset({('migrations', 0, 'rate')}) <= 0.0
    Row 3: frozenset({('migrations', 0, 'rate')}) <= 1.0
    Row 4: -frozenset({('demes', 1, 'start_time'), ('migrations', 0, 'start_time'), ('migrations', 1, 'start_time'), ('demes', 2, 'start_time'), ('demes', 0, 'epochs', 0, 'end_time')}) <= 0.0
    --------------------------------------------------

Modifying the constraints:
------------------------------------------
In addition to the constraints automatically derived from the construction of the demographic model, users may impose custom constraints to reflect specific biological assumptions or modeling choices.

A common example is the symmetry constraint on migration rates. This reflects the assumption that gene flow between two populations occurs at the same rate in both directions.

To enforce symmetric migration rates, we can add a new equality rule to the constraint matrices returned by ``constraints_for``.

Using the same IWM model, let's say we want to infer 3 parameters - the ancestral population size and the symmetric migration rate between P0 and P1. We start by obtaining the default constraints:

.. code-block:: python
    momi3_parameters = [
        frozenset({
            ("demes", 0, "epochs", 0, "end_size"),
            ("demes", 0, "epochs", 0, "start_size"),
        }), # Ancestral population size (index 0)
        ("migrations", 0, "rate"), # Rate of migration P0 to P1 (index 1)
        ("migrations", 1, "rate"), # Rate of migration P1 to P0 (index 2)
    ]

    constraint = constraints_for(et, *momi3_parameters)

With the code above, our constraint looks like this:

.. code-block:: python

    {'eq': (array([], shape=(0, 3), dtype=float64), array([], dtype=float64)),
    'ineq': (array([[-1., -0., -0.],
                    [-0., -1., -0.],
                    [ 0.,  1.,  0.],
                    [-0., -0., -1.],
                    [ 0.,  0.,  1.]]),
            array([0., 0., 1., 0., 1.]))
    }

As expected, there are no equality constraints, and the inequality constraints ensure nonnegative population size and migration rates bounded between [0,1].

Then, we can modify the constraint to enforce symmetry in migration rates using the ''modify_constraints_for_equality'' function. We provide the original ''constraint'' and the ''indices'' of the variables that we want to constraint.

.. code-block:: python
    from demesinfer.fit.util modify_constraints_for_equality
    new_constraint = modify_constraints_for_equality(constraint, [(1, 2)])
    print(new_constraint)

Sure enough, the updated constraint now includes the symmetry condition:

.. code-block:: python

    {'eq': (array([[ 0.,  1., -1.]]), array([0.])),
     'ineq': (array([[-1., -0., -0.],
             [-0., -1., -0.],
             [ 0.,  1.,  0.],
             [ 0.,  0., -1.]]),
      array([0., 0., 1., 0.]))}

If one wants to modify the inequalities, one can directly modify constraints. For example, if we want to constraint population size >= 2000., then we would do:

.. code-block:: python

    new_constraint["ineq"][1][0] = -2000.

Here we change the *first* parameter of the *second* element of the tuple (A_ineq, b_ineq).

Note that ``Frozenset`` parameters cannot be modified or directly removed, since they are derived from the demographic model structure. However, frozenset parameters disappear when the model no longer forces equality. For example, if a population’s size is not constant across an epoch (e.g., exponential growth), its start_size and end_size become separate variables instead of a single tied frozenset.

To show that, let's define a new demographic model where population size changes over time.

.. code-block:: python

    demo = msp.Demography()
    demo.add_population(name="anc", initial_size=5000)
    demo.add_population(name="P0", initial_size=5000, growth_rate=0.002)
    demo.add_population(name="P1", initial_size=5000, growth_rate=0.002)
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    demo.add_population_split(time=1000, derived=[f"P{i}" for i in range(2)], ancestral="anc")

This is a model where P0 and P1 grow exponentially from an initial size of 5000 at a rate of 0.002 per generation.

Rather than simulating, we directly examine parameters and constraints:

.. code-block:: python

    g = demo.to_demes()
    et = EventTree(g)
    et.variables

The output is:

.. code-block:: python
    
    [frozenset({('demes', 0, 'epochs', 0, 'end_size'),
            ('demes', 0, 'epochs', 0, 'start_size')}),
     frozenset({('demes', 1, 'epochs', 0, 'start_size')}),
     frozenset({('demes', 1, 'epochs', 0, 'end_size')}),
     frozenset({('demes', 2, 'epochs', 0, 'start_size')}),
     frozenset({('demes', 2, 'epochs', 0, 'end_size')}),
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

We can see that indeed, the population sizes for P0 and P1 are now treated as separate parameters (no longer in a frozenset), since they can differ due to exponential growth. The ancestral population size remains constant, so it is still grouped in a frozenset.

Correspondingly, the constraints will reflect this change. If you want to peek at all of the constraints, you can run:

.. code-block:: python

    constraints_for(et, *et.variables)

You would see that the start and end sizes for P0 and P1 are now independent variables without equality constraints tying them together.

Inference using SFS-based methods in momi3
------------------------------------------
momi3 provides a likelihood function based on the expected site frequency spectrum (SFS) under a demographic model. This makes it possible to perform demographic inference by optimizing model parameters to maximize the likelihood of the observed SFS.

As a first step, let’s focus on a single parameter: the migration rate from P0 to P1.

Using the simulated data and the original IWM model, we compute the observed allele frequency spectrum (AFS) from 4 haploid samples each of populations P0 and P1. We then initialize an `ExpectedSFS` object, which defaults to parameter values matching the input demographic model `demo`. To evaluate alternative parameter sets, we can compute the expected SFS by calling `ESFS(params)` with different parameter values as shown below.

.. code-block:: python

    from demesinfer.sfs import ExpectedSFS
    from demesinfer.loglik.sfs_loglik import sfs_loglik
    import jax
    import numpy as np
    
    param_key = frozenset({('migrations', 0, 'rate')})
    afs_samples = {"P0": 4, "P1": 4}
    afs = ts.allele_frequency_spectrum(
        sample_sets=[ts.samples([1])[0:4], ts.samples([2])[0:4]],
        span_normalise=False,
        polarised=True)
    ESFS = ExpectedSFS(demo.to_demes(), num_samples=afs_samples)
    
    @jax.value_and_grad
    def ll_at(val):
        params = {param_key: val}
        esfs = ESFS(params)
        return sfs_loglik(afs, esfs)
    
    loglik_value, loglik_grad = ll_at(0.0002)
    
Using JAX's automatic differentiation capabilities via the `@jax.value_and_grad` decorator, the `ll_at(val)` function simultaneously evaluates the log-likelihood and computes its gradient.

.. code-block:: python

    val = 0.0002
    print("Log-likelihood at rate =", val, "is", loglik_value)
    print("Gradient at rate =", val, "is", loglik_grad)

.. code-block:: python

    Log-likelihood at rate = 0.0002 is -0.00059234
    Gradient at rate = 0.0002 is -0.00460453

The values themselves don’t mean much in isolation, but they demonstrate how to call the likelihood and obtain its gradient. This is the foundation for parameter inference: we can now pass these functions to a numerical optimizer such as ``scipy.optimize.minimize`` to estimate the migration rate.

In the following examples, we will infer three types of parameters: population sizes, split times, and migration rates.

**Note**: Inference with large sample sizes using SFS may be slow. Consider reducing the number of samples when running locally. For reference, in all the examples below, I used a sample size of 10 diploids and ran them locally on a MacBook with an M2 chip. The runtime is roughly around 10-15 seconds. Simply bumping this number to 20 diploids results in a 5 minute runtime.

In the next section, to visually inspect how the likelihood changes (and assess reliability), we define a helper function ``plot_sfs_likelihood`` to plot the results. 

Estimating the ancestral population size
----------------------------------------

We first infer the size of the ancestral population ``anc``.  
With an initial guess of 4000, we evaluate the likelihood over a grid of values from 4000 to 6000:

.. code-block:: python

    import jax.numpy as jnp
    paths = {
        frozenset({
            ("demes", 0, "epochs", 0, "end_size"),
            ("demes", 0, "epochs", 0, "start_size"),
        }): 4000.,
    }
    afs_samples = {"P0": 20, "P1": 20}
    afs = ts.allele_frequency_spectrum(
            sample_sets=[ts.samples([1]), ts.samples([2])],
            span_normalise=False,
            polarised=True)

    vec_values = jnp.linspace(4000, 6000, 50)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: images/pop_size.png
   :alt: Ancestral population size inference
   :align: center

The negative log-likelihood is minimized around 4600, close to the true value of 5000.

Estimating the descendant population size
-----------------------------------------

Next, we infer the size of descendant population ``P0``.  
Again, starting from 4000, we search over values between 4000 and 6000:

.. code-block:: python

    import jax.numpy as jnp
    paths = {
        frozenset({
            ("demes", 1, "epochs", 0, "end_size"),
            ("demes", 1, "epochs", 0, "start_size"),
        }): 4000.,
    }
    vec_values = jnp.linspace(4000, 6000, 50)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: images/pop_size2.png
   :alt: Descendant population size inference
   :align: center

Here, the negative log-likelihood is minimized around 5500, close to the true value of 5000.

Estimating the split time
-------------------------

We then infer the split time between the ancestral population and its two descendants.  
This parameter is shared across multiple paths (two deme start times, one epoch end time, and two migration start times):

.. code-block:: python

    import jax.numpy as jnp
    paths = {
        frozenset({
            ("demes", 0, "epochs", 0, "end_time"),
            ("demes", 1, "start_time"),
            ("demes", 2, "start_time"),
            ("migrations", 0, "start_time"),
            ("migrations", 1, "start_time"),
        }): 4000.,
    }
    vec_values = jnp.linspace(500, 1500, 50)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: images/split_time.png
   :alt: Split time inference
   :align: center

The negative log-likelihood is minimized around 1000, correctly recovering the split time.

Estimating the migration rate
-----------------------------

Finally, we infer the migration rate between the two descendant populations:

.. code-block:: python

    import jax.numpy as jnp
    paths = {
        frozenset({("migrations", 0, "rate")}): 0.0001,
    }

    vec_values = jnp.linspace(0.00005, 0.0002, 10)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: images/migration_rate.png
   :alt: Migration rate inference
   :align: center

The negative log-likelihood is minimized around 0.00013, close to the true value of 0.0001.

Optimization with Poisson Likelihood
-----------------------------

So far, we have used the multinomial likelihood, which is the default in sfs_loglik when we haven’t provided a mutation rate theta; it conditions on the total number of segregating sites. An alternative is the Poisson likelihood, which models the absolute counts of mutations given the mutation rate theta and the sequence length.

This requires passing **BOTH** mutation rate ``theta`` and ``sequence_length`` into the likelihood function. These parameters depend on the species and the research itself. The setup is the same as before, but now we explicitly provide these parameters. Let's try to optimize the migration rate again, but using the Poisson likelihood this time.

.. code-block:: python

    import jax.numpy as jnp

    # The true values used in simulation were:
    theta = 1e-8
    sequence_length = 1e8

    # Example: estimating the migration rate with the Poisson likelihood
    paths = {
        frozenset({("migrations", 0, "rate")}): 0.0001,
    }

    vec_values = jnp.linspace(0.00005, 0.0002, 10)
    result = plot_sfs_likelihood(
        g, paths, vec_values,
        afs, afs_samples,
        theta=theta,
        sequence_length=sequence_length,
    )

.. image:: images/pois_migration.png
   :alt: Migration rate inference with Poisson likelihood
   :align: center

Poisson likelihood is also optimized near the real value of migration rate 0.0001. Compared to the multinomial likelihood, the Poisson can perform well if there is strong prior knowledge regarding the mutation rate. However, be aware that if estimates of the mutation rate are not accurate, then that can significantly impact the accuracy for the Poisson likelihood. 

Population size change example
-----------------------------
We now consider a more complex demographic model that includes population size changes and migration rate changes over time.

.. code-block:: python

    import msprime as msp
    import demes
    import demesdraw
    import numpy as np

    # Create demography object
    demo = msp.Demography()

    # Add populations
    demo.add_population(initial_size=4000, name="anc")
    demo.add_population(initial_size=500, name="P0", growth_rate=-np.log(3000 / 500)/66)
    demo.add_population(initial_size=500, name="P1", growth_rate=-np.log(3000 / 500)/66)
    demo.add_population(initial_size=100, name="P2", growth_rate=-np.log(3000 / 100)/66)

    # Set initial migration rate
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    demo.set_symmetric_migration_rate(populations=("P1", "P2"), rate=0.0001)


    # population size changes near 65–66 generations
    demo.add_population_parameters_change(
        time=65,
        initial_size=3000,  # Bottleneck: reduce to 1000 individuals
        population="P0",
        growth_rate=0
    )
    demo.add_population_parameters_change(
        time=65,
        initial_size=3000,  # Bottleneck: reduce to 1000 individuals
        population="P1",
        growth_rate=0
    )
    demo.add_population_parameters_change(
        time=66,
        initial_size=3000,  # Bottleneck: reduce to 1000 individuals
        population="P2",
        growth_rate=0
    )

    # Migration rate change changed to 0.001 AFTER 500 generation (going into the past)
    demo.add_migration_rate_change(
        time=66,
        rate=0.0005, 
        source="P0",
        dest="P1"
    )
    demo.add_migration_rate_change(
        time=66,
        rate=0.0005, 
        source="P1",
        dest="P0"
    )
    demo.add_migration_rate_change(
        time=66,
        rate=0.0005, 
        source="P1",
        dest="P2"
    )
    demo.add_migration_rate_change(
        time=66,
        rate=0.0005, 
        source="P2",
        dest="P1"
    )

    # THEN add the older events (population split at 1000)
    demo.add_population_split(time=5000, derived=["P0", "P1", "P2"], ancestral="anc")

    # Visualize the demography
    g = demo.to_demes()
    demesdraw.tubes(g, log_time=True)

.. image:: images/pop_size_change.png
    :alt: Population size change model visualization
    :align: center

**Note** The choice to use 65 (and 66) generations is intentional. In momi3, the event times that coincide exactly are treated as the same time identity and will be grouped into a single parameter (Check the notation section for more details). That’s useful when events truly share a time, but it can also merge parameters you’d prefer to optimize independently. The only way to avoid having frozensets forcefully constrain parameters to be equal is to modify the **construction** of the model. Offsetting one set of events to 65 generations and the others to 66 keeps them as distinct time variables.

You can inspect the parameters/constraints and see the effect using the same commands as before:

.. code-block:: python

    from demesinfer.constr import constraints_for, EventTree
    demo = g
    et = EventTree(demo)
    et.variables

Admixture example
-----------------------------
Another common demographic scenario of interest is admixture.

Here, we extend the simple IWM example to include four populations: one ancestral population (anc) and three contemporary populations (P0, P1, and ADMIX). We introduce an admixture event in which ADMIX is formed from P0 and P1 500 generations ago. At 1000 generations, P0 and P1 then merge back into the ancestral population.

An admixture event means that, going backwards in time, lineages from ADMIX are probabilistically reassigned to the source populations: in this case, with probability 0.4 to P0 and with probability 0.6 to P1. After the admixture time (500 generations ago), the ADMIX population becomes inactive.

.. code-block:: python

    demo = msp.Demography()
    demo.add_population(initial_size=5000, name="anc")
    demo.add_population(initial_size=5000, name="P0")
    demo.add_population(initial_size=5000, name="P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    tmp = [f"P{i}" for i in range(2)]
    
    demography = msp.Demography()
    demography.add_population(name="P0", initial_size=5000)
    demography.add_population(name="P1", initial_size=5000)
    demography.add_population(name="ADMIX", initial_size=1000)
    demography.add_population(name="anc", initial_size=5000)
    demography.add_admixture(
        time=500, derived="ADMIX", ancestral=["P0", "P1"], proportions=[0.4, 0.6])
    demography.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")


    g = demography.to_demes()
    demesdraw.tubes(g)

Again, we can visualize the demographic model using ``demesdraw``:

.. image:: images/pop_admixture.png
   :alt: Admixture model visualization
   :align: center

.. code-block:: python
    
    from demesinfer.event_tree import EventTree
    from demesinfer.constr import constraints_for

    et = EventTree(g)
    for v in et.variables:
        print(v)

In summary, the admixture event expands the parameter space by adding admixture proportions, and the constraints ensure that these proportions form a valid probability distribution.
