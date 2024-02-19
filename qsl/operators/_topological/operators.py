from typing import Tuple
from functools import partial

import jax
from jax import vmap, jit
import jax.numpy as jnp


from ...lattice import neighbors
from .base import TopoOperator


class Q(TopoOperator):
    '''
    Defines the off-diagonal operator, denoted X in Semeghini, Q in Verresen
    Applied on site i, it swaps sj<->sk if sj sk = -1 and flips si<->-si otherwise (for i,j,k nn)
    '''

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
        m = jnp.ones(x.shape[0]) #old -x[:,j]*x[:,k]

        # xj*xk = 1 if they are in the same state, case in which we flip i => x'i = -xi = -xi *(xj*xk)
        # otherwise it doesnt change it
        x_prime = x.at[:,i].set( -x[:,i]*x[:,j]*x[:,k] )

        # in all cases swap j and k, even if they have the same value
        x_prime = x_prime.at[:,j].set( x[:,k] )
        x_prime = x_prime.at[:,k].set( x[:,j] )
        
        
        return x_prime, m


class P(TopoOperator):
    '''
    Defines the off-diagonal operator, denoted Z in Semeghini, P in Verresen
    This operator is diagonal, so it always returns the same state, yet the amplitude of this is sj sk if it is applied on i (i,j,k nn)
    '''

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
        

        # m = sj*sk
        m = x[:,j]*x[:,k] # are j and k equal

        # diagonal op -> x_prime = x
        return x, m

class R(TopoOperator):
    '''
    Defines the off-diagonal operator, not used in Semeghini but should generate f-anyons on the lattice
    This is the product of Q0 P2 applied on multiple triangles, generalized on any site of the triangle. 
    '''

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

        # which case are we in : s1 s2 = -1 or s1 s2 = 1
        flag = x[:,j]*x[:,k]


        # if sj * sk = 1, flip i with amplitude -1
        def flip(x,i,j,k):
            return x.at[i].set( -x[i] )

        # is sj * sk = -1, swap sj and sk with amplitude 1
        def swap(x,i,j,k):
            Exchanged = x.at[j].set( x[k] )
            Exchanged = Exchanged.at[k].set( x[j] )
            return Exchanged

        # the final state is : i flipped if flag==1, jk swaped if flag ==-1
        eta = vmap(jax.lax.cond, in_axes=(0,None,None,0,None,None,None))(flag==1, flip, swap, x,i,j,k)

        # the amplitude is always -flag
        m = -x[:,i]*x[:,j] #old x[:,i]*x[:,k]

        return eta, m
