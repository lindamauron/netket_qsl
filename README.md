# Qunatum Spin Liquid simulation
This package allows for the simulation of the article "Variational Simulation of a topological spin liquid"

## Main classes : 
- **lattice** : used to describe the Kagome lattice of the experiment. Multiple shapes are possible (rectangular or ruby) with multiple boundary conditions for the rectangular-shaped ones. 
- **hilbert** : custom retricted hilbert space with perfect first neighbor blockade
- **operator** : all operators necessary, i.e. Rydberg hamiltonian as well as topological operators. 
- **frequencies** : callables of the frequency schedules one can use for Ω(t) and Δ(t). already implemented : Linear or Cubic schedule. 
- **models** : NQS models used for the time evolution. Mainly mean-field and jastrow ansatze. 
- **rules** : custom sampling rules (restricted means compatible with the restricted hilbert space).
- **driver** : mean-field driver for the analytical resolution of the TDVP. 

## On the side
Beside these main classes, there are a few utilities one can use : 
- **callbacks** to use during evolution
- **utils** with some often used utilities
- **toy_model** for models working specifically only on the lattice Torus(a,2,4) with restricted Hilbert space (where things can be computed exactly)

## Full example
```python
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import netket as nk
import netket.experimental as nkx

import flax.linen as nn
init = nn.initializers


import netket_qsl as qsl
folder = ''
# First, create your lattice
lattice = qsl.lattice.Ruby(3.9,extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4]) # lattice of Semeghini
N = lattice.N
hi = nk.hilbert.Spin(1/2, N) # standard hilbert space

# Now we define our variational model
ma = qsl.models.JMF_inv(jastrow_init=init.constant(0), mf_init=init.constant(1), lattice=lattice, n_neighbors=lattice.n_distances )
sa = nk.sampler.MetropolisSampler(hi, qsl.rules.TriangleRuleQ(), n_chains=72 )
vs = nk.vqs.MCState(sa, ma, n_samples_per_rank=1500, n_discard_per_chain=100 ) #, chunk_size=32)


# Define the Hamiltonian of the system
frequencies = qsl.frequencies.Cubic(2.5,1.4,-8,9.4)
H = qsl.operators.Hamiltionian(hi,lattice, frequencies)

# The observables
n_op = qsl.operators.r_density(hi,lattice)
P_op,Q_op,R_op = qsl.operators.TopoOps(hi,lattice,hex=0)
# only on one hexagon as a benchmark

obs = {}
obs['n'] = n_op
obs['P'] = P_op
obs['Q'] = Q_op

# The callbacks
cbs = []
if not sa.is_exact : 
    cbs.append( qsl.callbacks.callback_acc ) # acceptance if we have a MCMC sampler
cbs.append( qsl.callbacks.callback_omega_delta ) # writes down the value of the frequencies at each iteration
cbs.append( qsl.callbacks.callback_dimerprobs(lattice) ) # dimer probabilities


# The loggers 
logmf = nk.logging.JsonLog(folder+'MF', save_params=False)
logvs = qsl.logging.CompleteStateLog(folder+'states', save_every=10, tar=False)

T_MF = 0.2
step=1e-2
times = np.linspace(0.0, T_MF, np.rint(T_MF/step+1).astype(int), endpoint=True)


# Define the MF driver first
te_mf = qsl.driver.TDVP_MF(H,vs,t0=0.0, integrator=qsl.driver.RK4(1e-3))


te_mf.run(
    T=T_MF,
    out=[logmf,logvs],
    obs=obs,
    tstops=times,
    show_progress=True,
    callback=cbs,
)


logtdvp = nk.logging.JsonLog(folder+'TDVP', save_params=False)
# And once this is done, we do the normal t-VMC
te = nkx.TDVP(
    H,
    variational_state=vs,
    integrator=nkx.dynamics.RK4(dt=1e-2),
    t0=T_MF,
    qgt=nk.optimizer.qgt.QGTJacobianDense(diag_shift=0, holomorphic=True),
    error_norm="qgt",
    linear_solver=partial(nk.optimizer.solver.svd, rcond=1e-5 )
)


# perform the time-evolution saving the observable at every `tstop` time
te.run(
    T=frequencies.sweep_time-T_MF,
    out=[logtdvp,logvs],
    show_progress=True,
    obs=obs,
    #tstops=times,
    callback=cbs
)

```