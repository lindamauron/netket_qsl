import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import netket as nk
import netket.experimental as nkx

import jax.random as rnd
key = rnd.PRNGKey(12)

import flax.linen as nn
init = nn.initializers.normal()


import netket_qsl as qsl

# First, create your lattice
lattice = qsl.lattice.Torus(1.0, 2, 4) # toy model
#lattice = qsl.lattice.Ruby(3.9,extents_down=[3,4,5,6,7,6,5], extents_up=[4,5,6,7,6,5,4]) # lattice of Semeghini
#lattice = qsl.lattice.Square(3.9,8,12) # rectangular lattice with OBC for entropy
#lattice = qsl.lattice.OneTriangle(1.0) # debugging

N = lattice.N

# Corresponding hilbert space 
hi = qsl.hilbert.TriangleHilbertSpace(lattice) #restricted for small systems (can also be brought through sampler)
hi = nk.hilbert.Spin(1/2, N) # standard hilbert space
#hi.all_states()

# Now we define our variational model
init = nn.initializers
# chose anything in qsl.models, but for MF evolution, we need a model with an external mean-field (usuallly, JMF..)
ma = qsl.models.JMF_inv(jastrow_init=init.constant(0), mf_init=init.constant(1), lattice=lattice, n_neighbors=lattice.n_distances )
# sampler : 
#sa = nk.sampler.ExactSampler(hi)
sa = nk.sampler.MetropolisSampler(hi, qsl.rules.TriangleRuleQ(), n_chains=10 )
#sa = nk.sampler.MetropolisSampler(hi, qsl.rules.HexagonalRule(), n_chains=10 )
#sa = nk.sampler.MetropolisSampler(hi, qsl.rules.RestrictedMixedRule(lattice.hexagons, p_global=0.5), n_chains=10 )
# variational state
vs = nk.vqs.MCState(sa, ma, n_samples_per_rank=1500, n_discard_per_chain=0 ) #, chunk_size=32)


# Define the Hamiltonian of the system
frequencies = qsl.frequencies.Cubic(2.5,1.4,-8,9.4)
H = qsl.operators.Hamiltionian(hi,lattice, frequencies)

# The observables
n_op = qsl.operators.r_density(hi,lattice)
P_op,Q_op,R_op = qsl.operators.TopoOps(hi,lattice,hex=0)

obs = {}
obs['n'] = n_op
obs['P'] = P_op
obs['Q'] = Q_op
#obs['R'] = R

# The callbacks
cbs = []
if not sa.is_exact : 
    cbs.append( qsl.callbacks.callback_acc ) # acceptance if we have a MCMC sampler
cbs.append( qsl.callbacks.callback_omega_delta ) # writes down the value of the frequencies at each iteration
cbs.append( qsl.callbacks.callback_dimerprobs(lattice) ) # dimer probabilities


# The loggers 
logmf = nk.logging.JsonLog('ResultsMF', save_params=False)
logvs = qsl.logging.CompleteStateLog('Results', save_every=10, tar=False)

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



logtdvp = nk.logging.JsonLog('ResultsTDVP', save_params=False)
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
    T=1.0-T_MF,
    out=[logtdvp,logvs],
    show_progress=True,
    obs=obs,
    #tstops=times,
    callback=cbs
)



plt.figure(figsize=(20,4))

############################################## H ##############################################
plt.subplot(1,4,1)
plt.title('Energy')

plt.plot( logmf['Generator']['iters'], logmf['Generator']['Mean'], color='C0' )
plt.plot( logtdvp['Generator']['iters'], logtdvp['Generator']['Mean'], color='C0' )

plt.xlabel('Time')
plt.ylabel(r'$\langle \mathcal{H} \rangle$')

############################################## n ##############################################
plt.subplot(1,4,2)
plt.title('Mean rydberg occupation')

plt.plot( logmf['n']['iters'], logmf['n']['Mean'], color='C0' )
plt.plot( logtdvp['n']['iters'], logtdvp['n']['Mean'], color='C0' )

plt.xlabel('Time')
plt.ylabel(r'$\langle n \rangle$')
plt.legend()


############################################## Z ##############################################
plt.subplot(1,4,4)
plt.title('Topological Operators')

plt.plot( logmf['P']['iters'], logmf['P']['Mean'], color='C0' )
plt.plot( logmf['Q']['iters'], logmf['Q']['Mean'], color='C1' )

plt.plot( logtdvp['P']['iters'], logtdvp['P']['Mean'], color='C0' )
plt.plot( logtdvp['Q']['iters'], logtdvp['Q']['Mean'], color='C1' )


plt.xlabel('Time')
plt.ylabel(r'$\langle O \rangle$')
plt.legend(['P', 'Q'])

############################################ Dimers ############################################
plt.subplot(1,4,3)
plt.title('Monomers')

plt.plot( logmf['monomer']['iters'], logmf['monomer']['value'], color='C0' )
plt.plot( logmf['dimer']['iters'], logmf['dimer']['value'], color='C1' )
plt.plot( logmf['double dimer']['iters'], logmf['double dimer']['value'], color='C2' )


plt.plot( logtdvp['monomer']['iters'], logtdvp['monomer']['value'], color='C0' )
plt.plot( logtdvp['dimer']['iters'], logtdvp['dimer']['value'], color='C1' )
plt.plot( logtdvp['double dimer']['iters'], logtdvp['double dimer']['value'], color='C2' )


plt.xlabel('Time')
plt.ylabel(r'Probabilities')
plt.legend(['Monomer', 'Dimer', 'Double dimer'])

################################## General Figure and Saving ##################################

plt.tight_layout()
#plt.savefig(folder+f'time_evolution.png', dpi=200, bbox_inches='tight')
plt.show()