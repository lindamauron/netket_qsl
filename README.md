# Quantum Spin Liquid simulation
This package allows for the simulation of the article "Variational Simulation of a topological spin liquid". 
It currently runs under Netket 3.12. 


## Convention
We start by simply expliciting the convention we used in terms of basis, so that everything is clear. 

The Rydberg Hilbert space is spanned by the local states $| g\rangle, |r\rangle$, which we map to the "occupation" convention $|g\rangle = |0\rangle$ and $|r\rangle = |1\rangle$. 

Inspired by the notation in Semeghini et al., we use 
$$\hat \sigma^z = 1-2 \hat n = |g\rangle\langle g| - |r\rangle\langle r| = |0\rangle\langle 0| - |1\rangle\langle 1|.$$

It follows that $$\hat\sigma^y = i|1\rangle\langle 0| - i |0\rangle\langle 1| = i|r\rangle\langle g| - i |g\rangle\langle r|$$, and the $\sigma^x$ as usual. 
The annhililation opperators are defined as $$\hat\sigma^+ = (\hat\sigma^x+i\hat\sigma^y)/2 = |0\rangle\langle 1| = |g\rangle\langle r|$$ and $$\hat\sigma^- = (\hat\sigma^x-i\hat\sigma^y)/2 = |1\rangle\langle 0| = |r\rangle\langle g|$$ counter-intuitively. 

Netket's convention states that $\hat\sigma^z = |-1\rangle\langle -1| - |+1\rangle\langle +1|$, from which we identify $$|g\rangle = |0\rangle = |-1\rangle.$$
All the rest follows accordingly. Simply notice that the value $z$ in the vectors is defined such that $z=-1 \iff |\sigma\rangle = |g\rangle$. 


## Main classes : 
- **lattice** : used to describe the Kagome lattice of the experiment. Multiple shapes are possible (rectangular or ruby) with multiple boundary conditions for the rectangular-shaped ones. 
```python
import qsl
lattice = qsl.lattice.TwoTriangles(a)
lattice = qsl.lattice.Torus(a,2,4)
lattice = qsl.lattice.Square(a,3,6)
lattice = qsl.lattice.Ruby(3.9,extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4])
```
- **hilbert** : custom restricted hilbert space with perfect first neighbor blockade
- **frequencies** : callables of the frequency schedules one can use for Ω(t) and Δ(t). already implemented : Linear or Cubic schedule. 
```python
f = qsl.frequencies.Cubic(sweep_time,Omegaf,Deltamin,Deltamax)
f = qsl.frequencies.Linear(sweep_time,Omegaf,Deltamin,Deltamax)
```
- **operator** : all operators necessary, i.e. Rydberg hamiltonian as well as topological operators and basic Pauli matrices.  
```python
import qsl
n_op = r_density(hi, lattice)
P_op, Q_op, R_op = TopoOps(hi, lattice, hex=0) # all topological operators for the same contour on the lattice
P = -2*qsl.operators.P(hi,sites)*qsl.operators.P(hi,other_sites) #topo operators have basic mathematical compatibilities

H = qsl.operators.Rydberg_Hamiltonian(hilbert,lattice,frequencies,Rb,Rcut)
# you can then call the Hamiltonain at any time : 
H(t)

# or obtain the Hamiltonian for a certain Delta value :
H.of_delta(d)

# or have the separate operators of the Hamiltonian (without time dependens) in LocalOperator or sparse matrices
H.operators(), H.for_sparse()
```
- **observables** : to compute the Renyi2 entanglement entropy of the system.
```
S2 = qsl.observables.Renyi2EntanglementEntropy(hilbert,partition,n_boots,seed)
vs.expect(S2)
``` 
You can either evaluate the error through error propagation (not recommended) by setting n_boots=None,0,1 or through bootstrapping by choosing a finite number of bootstraps. 

- **models** : NQS models used for the time evolution. Mainly mean-field and jastrow ansatze. 
- **rules** : custom sampling rules (restricted means compatible with the restricted hilbert space).
- **driver** : mean-field driver for the analytical resolution of the TDVP. 

## On the other side
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
ma = qsl.models.JMF_inv(jastrow_init=init.constant(0), mf_init=init.constant(1), lattice=lattice )
sa = nk.sampler.MetropolisSampler(hi, qsl.rules.TriangleRuleQ(), n_chains=72 )
vs = nk.vqs.MCState(sa, ma, n_samples_per_rank=1500, n_discard_per_chain=100 ) #, chunk_size=32)


# Define the Hamiltonian of the system
frequencies = qsl.frequencies.Cubic(2.5,1.4,-8,9.4)
H = qsl.operators.RydbergHamiltonian(hi,lattice, frequencies)

# The observables
n_op = qsl.operators.r_density(hi,lattice)
P_op,Q_op,R_op = qsl.operators.TopoOps(hi,lattice,hex=0)
# only on one hexagon as a benchmark

obs = {}
obs['n'] = n_op
obs['P'] = P_op
obs['Q'] = Q_op
DimerProbs = qsl.observables.DimerProbabilities(hi,lattice,bulk=False)

# The callbacks
cbs = []
if not sa.is_exact : 
    cbs.append( qsl.callbacks.callback_acc ) # acceptance if we have a MCMC sampler
cbs.append( qsl.callbacks.callback_omega_delta ) # writes down the value of the frequencies at each iteration
cbs.append( qsl.callbacks.callback_dimerprobs(DimerProbs) ) # dimer probabilities

## The callbacks
cbs = []
if not sa.is_exact : 
    cbs.append( qsl.callbacks.callback_acc ) # acceptance if we have a MCMC sampler
cbs.append( qsl.callbacks.callback_omega_delta ) # writes down the value of the frequencies at each iteration
cbs.append( qsl.callbacks.callback_dimerprobs(DimerProbs) ) # dimer probabilities

T_MF = 0.2
step=1e-2

# Define the MF driver first
te_mf = qsl.driver.TDVP_MF(H,vs,t0=0.0, integrator=qsl.driver.RK4(1e-3))
## perform the time-evolution saving the observable at every `tstop` time
te_mf.run(
    T=T_MF,
    out=folder,
    obs=obs,
    # tstops=times,
    show_progress=True,
    callback=cbs,
)

## The loggers
logmf = json.load(open(folder+'MF.log'))
logtdvp = nk.logging.JsonLog(folder+'TDVP', save_params=False)
logvs = nk.logging.StateLog(folder+'_states', save_every=1, tar=False)

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
    T=T-T_MF,
    out=[logtdvp,logvs],
    show_progress=True,
    obs=obs,
    callback=cbs
)


```