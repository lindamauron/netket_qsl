'''
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (4, 3)

import numpy as np

import netket as nk

import jax.random as rnd
key = rnd.PRNGKey(12)

import flax.linen as nn
init = nn.initializers.normal()


import Lattice
from TopologicalOperators import P,Q,R



lattice = Lattice.Torus(1.0, 2, 4)
N = lattice.N

from Hilbert import TriangleHilbertSpace
hi = TriangleHilbertSpace(lattice)
#hi.all_states()


from Models import JasMF_trans
# Model
init = nn.initializers.constant
ma = JasMF_trans(jastrow_init=init(0), mf_init=init(1), lattice=lattice, n_neighbors=np.max(lattice.neighbors_distances)+1 )
from Rules import TriangleRuleQ
sa = nk.sampler.MetropolisSampler(hi, TriangleRuleQ(), n_chains=3 )
# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples_per_rank=15, n_discard_per_chain=0 ) #, chunk_size=32)




from ZOperators import Hamiltionian
H = Hamiltionian(hi,lattice)





from qsl.driver import TDVP_MF, RK4

from ZOperators import r_density
n = r_density(hi,lattice)

t = 0.0
T = 0.5
step = 1e-2
times = np.linspace(t, t+T, np.rint(T/step+1).astype(int), endpoint=True)
log = nk.logging.JsonLog('test', save_params=False)

from qsl.callbacks import callback_dimerprobs_MF
d = TDVP_MF(H,vs,t0=t, integrator=RK4(1e-3))
print(d.info())
d.run(T,out=log, obs={'n':n, 'P':P(hi,0)}, tstops=times, callback=callback_dimerprobs_MF(lattice))

import matplotlib.pyplot as plt
print(log['Generator']['iters'])

plt.figure()
plt.plot(log['Generator']['iters'], log['Generator']['Mean'])
plt.show()

plt.figure()
plt.plot(log['n']['iters'], log['n']['Mean'] )
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
import flax.linen as nn
import netket as nk
import netket_qsl as qsl
import jax.numpy as jnp
import flax
import jax
lattice = qsl.lattice.Torus(1.0,2,4)
N = lattice.N

hi = qsl.hilbert.TriangleHilbertSpace(lattice)

# Model
init = nn.initializers.constant
ma = qsl.models.JMF_inv(jastrow_init=init(0), mf_init=init(1), lattice=lattice, n_neighbors=lattice.n_distances )


# Sampler
n_chains = 48
sa = nk.sampler.MetropolisSampler(hi, qsl.rules.TriangleRuleQ(), n_chains=n_chains)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=n_chains*15, n_discard_per_chain=150, seed=13  ) #, chunk_size=32)
with open(f'../QSL/Entropy/PBC/Lx=2_Ly=4/Q_states/20.mpack', 'rb') as file:
    variables = flax.serialization.from_bytes(vs.variables, file.read())
vs.parameters = {'W':jnp.array(variables['params']['W']), 'ϕ':jnp.array(variables['params']['ϕ'])}
print(vs.samples.shape)
# import netket_fidelity as nkf
# S = nkf.Renyi2EntanglementEntropy(hi,np.arange(0,N//2))


# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# if rank==0:
#     print( vs.expect(S) )


log_data = {}

Ps = qsl.observables.DimerProbabilities(hi,lattice)
print(Ps)

ps = vs.expect(Ps)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
    print(ps)

total = 0
for p in ps:
    total += ps[p].mean

if rank==0:
    print(total)
log_data['probs'] = ps

# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# if rank==0:
#     print(log_data)
#     plt.figure(figsize=(6,3))

#     plt.subplot(1,2,1)
#     plt.hist(f.real)
#     plt.axvline(log_data['S'].mean.real, ls='-', color='k')
#     plt.axvline(log_data['S'].mean.real+np.sqrt(log_data['S'].variance), ls=':', color='k')
#     plt.axvline(log_data['S'].mean.real-np.sqrt(log_data['S'].variance), ls=':', color='k')
#     plt.xlabel('real part of S_2')
#     plt.ylabel('counts')

#     plt.subplot(1,2,2)
#     plt.hist(f.imag)
#     plt.xlabel('imag part of S_2')
#     plt.ylabel('counts')

#     plt.tight_layout()
#     plt.show()

# from zak import callback_entropy
# cb = callback_entropy(lattice,sites,13)

# log_data = {}
# cb(jax.random.PRNGKey(12),vs,log_data)
# print(log_data)

S = qsl.observables.Renyi2EntanglementEntropy(hi,[0,1,2])

log_data['S'] = vs.expect(S)


if rank==0:
    print(log_data)