from typing import Tuple
from functools import partial

import numpy as np
import jax
from jax import vmap, jit
import jax.numpy as jnp
import warnings

from typing import Union, Optional
from netket.utils.types import Array, ArrayLike, DType
from netket.hilbert import Spin as _SpinHilbert

from ...lattice._utils import neighbors
from .base import TopoOperator


class QX(TopoOperator):
    '''
    Defines the off-diagonal operator, denoted X in Semeghini, Q in Verresen, rotated in the X basis
    '''
    def __init__(self, hilbert:_SpinHilbert, sites:Union[int,Array,ArrayLike], scalar=1.0):
        warnings.warn('The X-basis classes are not updated', DeprecationWarning)
        super().__init__(hilbert,sites,scalar)
        
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

        # which case are we in : x1 x2 x3 = -1 or x1 x2 x3 = 1
        flag = x[:,i]*x[:,j]*x[:,k]


        # if xi * xj * xk = -1, do nothing
        def nothing(x,i,j,k):
            return x

        # if xi * xj * xk = 1, flip j and k
        def flip_both(x,i,j,k):
            Exchanged = x.at[j].set( -x[j] )
            Exchanged = Exchanged.at[k].set( -x[k] )
            return Exchanged

        # the final state
        eta = vmap(jax.lax.cond, in_axes=(0,None,None,0,None,None,None))(flag==-1, nothing, flip_both, x,i,j,k)

        # the amplitude is always the opposite of xi
        m = -x[:,i]

        return eta, m

class PX(TopoOperator):
    '''
    Defines the off-diagonal operator, denoted Z in Semeghini, P in Verresen, rotated in the X basis
    This operator is diagonal, so it always returns the same state, yet the amplitude of this is sj sk if it is applied on i (i,j,k nn)
    '''
    def __init__(self, hilbert:_SpinHilbert, sites:Union[int,Array,ArrayLike], scalar=1.0):
        warnings.warn('The X-basis classes are not updated', DeprecationWarning)
        super().__init__(hilbert,sites,scalar)


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
        

        # m = 1
        m = jnp.ones(x.shape[0])

        # flips spin j and k in all cases
        eta = x.at[:,j].set( -x[:,j] )
        eta = eta.at[:,k].set( -x[:,k] )

        return eta, m

class RX(TopoOperator):
    '''
    Defines the off-diagonal operator, not used in Semeghini but should generate f-anyons on the lattice
    This is the product of Q0 P2 applied on multiple triangles, generalized on any site of the triangle, rotated in the X basis
    '''

    def __init__(self, hilbert:_SpinHilbert, sites:Union[int,Array,ArrayLike], scalar=1.0):
        warnings.warn('The X-basis classes are not updated', DeprecationWarning)
        super().__init__(hilbert,sites,scalar)


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

        # which case are we in : x1 x2 x3 = -1 or x1 x2 x3 = 1
        flag = x[:,i]*x[:,j]*x[:,k]

        # if flag, flip sk
        def flipk(x,i,j,k):
            return x.at[k].set( -x[k] )

        # else, flip sj
        def flipj(x,i,j,k):
            return x.at[j].set( -x[j] )



        # intermediat state
        eta = vmap(jax.lax.cond, in_axes=(0,None,None,0,None,None,None))(flag==1, flipk, flipj, x,i,j,k)

        #in all cases, flip xi
        eta = eta.at[:,i].set( -eta[:,i] )

        # the amplitude is always -x0
        m = -x[:,i]

        return eta, m
