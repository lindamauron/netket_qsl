import numpy as np

from typing import Any
from netket.utils.types import NNInitFunc
import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn


#######################################################################################################################
####################################################### Models ########################################################
#######################################################################################################################


class Jas12(nn.Module):
    '''
    log \psi(z) = \sum_ij z_i W_ij z_j + \sum_i h_i z_i
    '''
    
    param_dtype : Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init : NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the 2 body matrix."""

    field_init : NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the field."""

    @nn.compact
    def __call__(self, x):
        '''
        x : (Ns,N)
        '''
        N = x.shape[-1]

        # Define the two variational parameters W and h
        W = self.param(
            "W", self.kernel_init, (N,N), self.param_dtype
        )
        h = self.param(
            'h', self.field_init, (N,), self.param_dtype
        )

        # compute the nearest-neighbor correlations
        corr1=jnp.einsum( '...i,ij,...j',x,W,x )

        # compute the field excitations
        field = jnp.einsum('i,...i',h,x)

        # sum the output
        return corr1 + field
    

class Jas1(nn.Module):
    '''
    log \psi(z) = \sum_i h_i z_i
    '''
    param_dtype : Any = jnp.complex128
    """The dtype of the weights."""

    init_fn : NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the field."""


    @nn.compact
    def __call__(self, x):
        N = x.shape[-1]
        h = self.param('h', self.init_fn, (N,), self.param_dtype)

        return jnp.einsum('i,...i',h,x)
   


