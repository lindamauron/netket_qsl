import jax.numpy as jnp
from jax._src import dtypes

import flax.linen as nn
from typing import Any
from ..lattice import Kagome as _Kagome
from ._jmfs import Psi_MF


def custom_init(z,infinity,dtype=jnp.float_):
    r'''
    Custom initializer for the φ parameters to match the RVB state
    z : 2nd and 3rd neighbors connectivity (N,) : z_i = \sum_j ( delta_{dij,2} + delta_{dij,3} )
    infinity : number used as the infinity limit for the model
    '''
    def init(key,shape,dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)

        phi = jnp.ones(shape,dtype=dtype)
        phi = phi.at[:,0].set( jnp.exp(+infinity*(z-2)/2) )
        phi = phi.at[:,1].set( jnp.exp(-infinity*(z-2)/2) )

        return phi
    return init


class RVB(nn.Module):
    r'''
    log \psi(z) = W/2 \sum_ij z_i z_j \chi(d_{ij}) + \sum_i \log( \phi(zi) )
    '''
    lattice : _Kagome
    """The lattice on which the model acts."""

    infinity : float = 1e2
    """Number used intead of infinity to get to the limit."""

    param_dtype : Any = jnp.float64
    """The dtype of the weights."""

    def setup(self):
        '''
        Initializes the model correctly, defining the chi matrix and the parameters.
        '''
        N = self.lattice.N

        self.W = self.param(
            'W', nn.initializers.constant(-self.infinity), (1,), self.param_dtype
        )


        chi = jnp.zeros((N,N))
        self.chi = chi.at[ (self.lattice.neighbors_distances==2) + (self.lattice.neighbors_distances==3) ].set(1)

        
        z = self.chi.sum(-1)
        self.phi = self.param(
            'ϕ', custom_init(z=z,infinity=self.infinity), (N,2), self.param_dtype
        )


    @nn.compact
    def __call__(self, x):
        '''
        x : (Ns,N)
        '''
        # compute the nearest-neighbor correlations
        corr = self.W/2 * jnp.einsum( '...i,ij,...j',x,self.chi,x )

        mf = Psi_MF(jnp.array(self.phi), x)
        
        return corr + jnp.sum(jnp.log(mf), axis=-1)
