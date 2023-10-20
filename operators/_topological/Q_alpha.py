from typing import Tuple
from functools import partial

import numpy as np
import jax
from jax import vmap, jit
import jax.numpy as jnp

from netket.utils.types import DType as _DType

from QSL.lattice import neighbors
from .base import TopoOperator


class Q_alpha(TopoOperator):
    '''
    Defines the off-diagonal operator, denoted X in Semeghini, Q in Verresen with arbitrary phase factor exp(i alpha)
    Applied on site i, it swaps sj<->sk if sj sk = -1 and flips si<->-si otherwise (for i,j,k nn)
    '''
    def __init__(self, hilbert, sites, alpha=0.0):
        super().__init__(hilbert, sites)
        self.q = jnp.exp(1j*alpha)

    @partial(jit, static_argnums=0)
    def _conn_one_triangle(self, x:jnp.ndarray, i:int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        finds the connected element and matrix element of a single element x for an operator acting on site i
        x : flat sample (Ns,N)
        i : single site on which the operator is acting

        return : σ (Ns,N), <x|O|σ> (Ns,)
        '''
        # Define the nn
        j, k = neighbors(i)

        x = jnp.array(x)

        # signs depends on wether j,k are excited
        m = (1-x[:,j]*x[:,k])/2 + (1+x[:,j]*x[:,k])/2*self.q**x[:,i]

        # xj*xk = 1 if they are in the same state, case in which we flip i => x'i = -xi = -xi *(xj*xk)
        # otherwise it doesnt change it
        x_prime = x.at[:,i].set( -x[:,i]*x[:,j]*x[:,k] )

        # in all cases swap j and k, even if they have the same value
        x_prime = x_prime.at[:,j].set( x[:,k] )
        x_prime = x_prime.at[:,k].set( x[:,j] )
        
        
        return x_prime, m

