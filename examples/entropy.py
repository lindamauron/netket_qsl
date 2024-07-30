import numpy as np

import jax.random as rnd
key = rnd.PRNGKey(12)

import netket as nk
nk.config.netket_experimental_fft_autocorrelation = True

import flax.linen as nn
import flax

import qsl as qsl
from qsl.observables import Renyi2EntanglementEntropy as Renyi2

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import sys
T = 2.5
folder = f'QSL/TDVP/Q'

lattice = qsl.lattice.Ruby(3.9)

times = np.array([2.5,2.45,2.4,2.35,2.3,2.2,2.1,2.])

offset = 0.2/2.5*T
steps = np.rint((times-offset)*100).astype(int)

N = lattice.N
hi = nk.hilbert.Spin(s=1/2, N=N)

# Model
init = nn.initializers.constant
ma = qsl.models.JMF_inv(jastrow_init=init(0), mf_init=init(1), lattice=lattice)

# Sampler
sa = nk.sampler.MetropolisSampler(hi, qsl.rules.TriangleRuleQ(), n_chains=36*2*10, n_sweeps=10*N )


# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples_per_rank=10*300, n_discard_per_chain=10) #, chunk_size=131072 )

# Define observables
cbs = []
obs = {}

# Ruby lattice
site0A = np.array([87,105])
site0B = np.array([108,129])
site0C = np.array([126,147])


G = qsl.observables.Gamma(hi,
                          np.array([site0A, site0A+1, site0A+2], dtype=int).reshape(-1),
                          np.array([site0B, site0B+1, site0B+2], dtype=int).reshape(-1),
                          np.array([site0C, site0C+1, site0C+2], dtype=int).reshape(-1),
                          n_boots=1024,chunk_post=512
                          )

cb = qsl.observables.callback_gamma_boots(G,folder+'ABC/',True)


log = nk.logging.JsonLog(folder+'Q_gamma_1kboot', save_params=False)
log1b = nk.logging.JsonLog(folder+'Q_gamma_noboot', save_params=False)

for i in steps:
    print(i)
    time = i/100 + offset

    with open(folder+f'_states/{i}.mpack', 'rb') as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
    vs.sample()

    log_data = {}
    log_data1b = {}

    cb(time,(log_data,log_data1b), vs)
    log(time, log_data, vs)
    log1b(time, log_data1b, vs)

    print('done')
