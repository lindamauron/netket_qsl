import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
from ..hilbert import TriangleHilbertSpace
from ..lattice import Torus
from ..operators import dimer_probs as _local_probs
import flax.linen as nn
from netket.utils.types import DType, Array, NNInitFunc
from scipy.sparse import csr_array

import os.path as path
text_file_path = path.dirname(path.abspath(__file__)) + '/indices.npy'
indices = np.load(text_file_path)
N=24

lattice = Torus(1.0,2,4)
hi = TriangleHilbertSpace(lattice)


def to_full_hilbert(psi):
    return csr_array( (psi, (indices, 0*indices)) , shape=(N,1))


restr_basis = hi.all_states()

def dimer_probs(lattice, psi):
    def pi(s,u):
        ps = _local_probs(lattice, s)
        return ps * u.conj()*u

    ps = jnp.mean( vmap(pi, in_axes=(0,0), out_axes=0)(restr_basis, psi), axis=0 )
    return ps/ps.sum()