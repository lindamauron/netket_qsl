import numpy as np 

from functools import partial

from netket.hilbert.custom_hilbert import CustomHilbert
from netket.utils.dispatch import dispatch

import jax 
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, vmap

from typing import Union
from ..lattice import Kagome as _Kagome

class TriangleHilbertSpace(CustomHilbert):
    '''
    Custom Hilbert space on which there is at most one excited state per triangle. Of course, this only works in the Z basis. 
    For this, one needs a new random_state, already in the subspace. 

    Note : we did not define the local rule on this space, the ExactSampler does work though. 
    For a local sampler, use MetropolisSampler with TriangleRule or RestrictedMixedRule to resepct the constraints.
    For unknown reason, it did not work to implement a new local rule, but TriangleRule does the work perfectly.

    On this subspace, th σ^x,σ^y operators have been modified in order to stay in the subspace. All components outside the subspace were set to zero. 
    '''

    def __init__(self, lattice:Union[_Kagome,None] = None, N:Union[int,None] = 3):
        '''
        Instantiates the Hilbert space
        lattice : lattice which we want the hilbert space to represent
                  one needs it in order to have the information about the triangles
        N: number of sites in the Hilbert space
                  if lattice is not given, we extrapolate the triangles through the number of sites
        '''
        if lattice:
            self.triangles = jnp.array([t['atoms'] for t in lattice.triangles])
            N = lattice.N

        else:
            if not N:
                N = 3
            if N%3!=0:
                raise ValueError(f'N must be a multiple of 3, instead got value {N} = 3*{N//3}+{N%3}')
            n_triangles = 3*jnp.arange(N//3)
            self.triangles = jnp.array([n_triangles, n_triangles+1, n_triangles+2]).T
            
        super().__init__([-1.0, 1.0], N, vmap(self.constraint_fn) )


        
        

    def constraint_fn(self, state):
        '''
        Indicates wether the state respects the constraint or not
        state : single sample to test (N,)

        return: boolean telling if the state is in the restricted space or not
        '''
        # number of excitation per triangle
        occupancy = -jnp.array([jnp.sum(state[...,t], axis=-1) for t in self.triangles])

        # number of triangles with 2,3 excited states
        n2 = jnp.count_nonzero((occupancy+1)==0, axis=-1)
        n3 = jnp.count_nonzero((occupancy+3)==0, axis=-1)

        # flag saying if there is at most 2,3 excitations one at least one triangle
        n2_flag = ( 1-n2.astype(bool) ).astype(bool)
        n3_flag = ( 1-n3.astype(bool) ).astype(bool)

        # multiply the flags to fulfill both conditions
        return n2_flag*n3_flag
    
    def __repr__(self):
        return f'TriangleHilbertSpace(N={self.size}'+', basis={|ggg>,|rgg>,|grg>,|ggr>}^N/3)'
    

@partial(jit, static_argnums=1)
def _mask(key, n_triangles):
    '''
    Generates the mask of where to put excited states on the lattice in order to have at most one per triangle
    key : 
    n_triangles : 

    return : 
    '''
    key, _ = jax.random.split(key, 2)

    # equal possibility to have {|rgg>, |grg>, |ggr>, |ggg>} = {0,1,2,3}
    indices = jax.random.choice( key, 4, shape=(n_triangles,) )

    def _one_tri(i):
        triangle = jnp.zeros( 3, dtype=bool )

        def swap(i):
            return triangle.at[i].set(True)
        def nothing(i):
            return triangle

        return jax.lax.cond(i==3, nothing, swap, i)


    return vmap(_one_tri)(indices).reshape(-1)


@dispatch
def random_state(hilb:TriangleHilbertSpace, key, batches: int, dtype):

    N = int(hilb.size)

    # start with all ground states
    σ = -np.ones( (batches, N) )


    keys = jax.random.split( key, batches )

    # for each triangle, find the mask (i.e. equal possibility to have |ggg>, |rgg>, |grg>, |ggr>)
    masks = vmap(_mask, in_axes=(0,None))(keys,int(N/3))

    σ[masks] = 1 

    # return the correctly shaped chain
    return σ
    
