import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import netket as nk
import netket.experimental as nkx

import jax.random as rnd
key = rnd.PRNGKey(12)

import flax.linen as nn
init = nn.initializers
import json


import sys
sys.path.append("..")

import qsl

folder = 'samplers/'

## First, create your lattice
lattice = qsl.lattice.Torus(1.0, 2, 4) # toy model

N = lattice.N

## Corresponding hilbert space 
#hi = qsl.hilbert.TriangleHilbertSpace(lattice) #restricted for small systems (can also be brought through sampler)
hi = nk.hilbert.Spin(1/2, N) # standard hilbert space


## Now we define our variational model
## chose anything in qsl.models, but for MF evolution, we need a model with an external mean-field (usuallly, JMF..)
ma = qsl.models.JMF_inv(jastrow_init=init.constant(0), mf_init=init.constant(1), lattice=lattice )
## sampler : 
#sa = nk.sampler.ExactSampler(hi)
# rule = qsl.rules.TriangleRuleQ()
# rule = qsl.rules.HexagonalRule(lattice=lattice)
rule = qsl.rules.RestrictedMixedRule(lattice=lattice, probs=[0.1,0.9])
sa = nk.sampler.MetropolisSampler(hi, rule, n_chains=12 )
## variational state
vs = nk.vqs.MCState(sa, ma, n_samples_per_rank=1500, n_discard_per_chain=0 ) #, chunk_size=32)


## Define the Hamiltonian of the system
T = 2.5
frequencies = qsl.frequencies.Cubic(T,1.4,-8,9.4)
H = qsl.operators.Rydberg_Hamiltionian(hi,lattice, frequencies)


## The callbacks
cbs = []
if not sa.is_exact : 
    cbs.append( qsl.callbacks.callback_acc ) # acceptance if we have a MCMC sampler


T_MF = 0.2
step=1e-2

# Define the MF driver first
te_mf = qsl.driver.TDVP_MF(H,vs,t0=0.0, integrator=qsl.driver.RK4(1e-3))
## perform the time-evolution saving the observable at every `tstop` time
te_mf.run(
    T=T_MF,
    out=folder,
    show_progress=False,
)

## The loggers
logtdvp = nk.logging.JsonLog(folder+'TDVP', save_params=False)

## And once this is done, we do the normal t-VMC
te = nkx.TDVP(
    H,
    variational_state=vs,
    integrator=nkx.dynamics.RK4(dt=1e-2),
    t0=T_MF,
    qgt=nk.optimizer.qgt.QGTJacobianDense(diag_shift=0, holomorphic=True),
    error_norm="qgt",
    linear_solver=partial(nk.optimizer.solver.svd, rcond=1e-5 )
)
## perform the time-evolution saving the observable at every `tstop` time
te.run(
    T=1.0,
    out=logtdvp,
    show_progress=True,
    #tstops=times,
    callback=cbs
)

qsl.utils.mpi_print(logtdvp['acc']['value'])