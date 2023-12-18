import jax.numpy as jnp
import numpy as np 

import netket as nk
from netket.hilbert import Spin as _SpinHilbert
from ...lattice import Kagome as _Kagome


class DimerProbabilities:
    r"""
    Observable able to compute the probabilities to find m dimers on the vertices of the lattice. Since each vertex can have up to 4 excitations, we define things as follow:
    - p0 = monomer
    - p1 = dimer
    - p2 = double dimer
    - p3 = triple dimer
    - p4 - quadruple dimer

    This implementation is compatible with netket's built-in expectation value. 
    """
    def __init__(self,hilbert:_SpinHilbert,lattice:_Kagome, bulk:bool=False):
        '''
        hilbert: hilbert space on which to compute qunatities
        lattice: lattice of which vertice's we want the occupation
            the lattice should have a container s.t. lattice.vertices[k]['atoms'] = [atoms connected to vertex k]
        bulk: bool indicating whether the probabilities are computed only inside the bulk or not
            if bulk==True, the lattice should have a container lattice.non_border = [vertices inside the bulk]
        '''
        self._hilbert = hilbert
        self._lattice = lattice
        if not bulk:
            self.v = [np.array(v['atoms']) for v in lattice.vertices]
        else:
            self.v = [np.array(v['atoms']) for v in lattice.vertices[lattice.non_border] ]

        self.n = len(self.v)


    @property
    def hilbert(self):
        r"""The hilbert space associated to this observable."""
        return self._hilbert
    
    @property
    def lattice(self):
        r"""The lattice associated to this observable."""
        return self._lattice

    def __repr__(self):
        return f"DimerProbabilities(hilbert={self.hilbert}, lattice={self.lattice})"
    