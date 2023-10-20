import numpy as np 

from typing import Tuple, Optional
from functools import partial
import abc

import netket as nk
from netket.utils.types import PyTree, PRNGKeyT
from netket import sampler

import jax 
import jax.numpy as jnp
from jax import jit, vmap
from flax import linen as nn


#################################################################################################################################
@partial(jit, static_argnums=1)
def _mask(key, n_triangles):
    '''
    Generates the mask of where to put excited states on the lattice in order to have at most one per triangle
    key : random key
    n_triangles : number of triangles

    return : boolean mask of where to put excited states of shape (N,)
    '''
    key, _ = jax.random.split(key, 2)

    # indices {0,1,2,3} = {|rgg>, |grg>, |ggr>, |ggg>}
    indices = jax.random.choice( key, 4, shape=(n_triangles,) )

    # Generates the mask of one triangle (3,)
    def _one_tri(i):
        triangle = jnp.zeros( 3, dtype=bool )

        def flip(i):
            # excites spin i
            return triangle.at[i].set(True)
        def nothing(i):
            # excites no spin
            return triangle

        return jax.lax.cond(i==3, nothing, flip, i)

    # does the mask on all the triangles
    return vmap(_one_tri)(indices).reshape(-1) #(N,)

@nk.utils.struct.dataclass
class RestrictedRule(nk.sampler.rules.MetropolisRule):
    '''
    A super class of rules working on the restricted Hilbert space \{ |000>, |100>, |010>, |001> \}.
    For those samplers to work properly, one needs to redefine random_state, which is done here. 
    Then, each subclass has to define the transition rule properly in that subspace. 
    '''
    
    def random_state(self, sampler: "sampler.MetropolisSampler", machine: nn.Module, params: PyTree, sampler_state: "sampler.SamplerState", key: PRNGKeyT) -> jnp.ndarray:
        '''
        Generates a batch of random spin chains verifying only one excited state by triangle

        sampler: The Metropolis sampler.
        machine: A Flax module with the forward pass of the log-pdf.
        params: The PyTree of parameters of the model.
        sampler_state: The current state of the sampler. Should not modify it.
        key: The PRNGKey to use to generate the random state.
        
        returns : σ the batch of spin chains
        '''                            
        size = sampler.n_batches

        Ns = int(np.prod(size))
        N = int(sampler.hilbert.size)

        # start with all ground states
        σ = -np.ones( (Ns, N) )

        # Random keys
        keys = jax.random.split( key, Ns )
        
        # masks on where to randomly put excited states respecting the constraint
        masks = vmap(_mask, in_axes=(0,None))(keys,int(N/3)) #(Ns,N)
        σ[masks] = 1 

        # return the correctly shaped chain
        return σ
    
    @abc.abstractmethod
    def transition(self, sampler: "sampler.MetropolisSampler", machine: nn.Module, params: PyTree, sampler_state: "sampler.SamplerState", key: PRNGKeyT, σ: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        r'''
        Transition rule which has to be defined in all subclasses. It proposes a new configuration set of configurations $\sigma'$ starting from the current
        chain configurations :math:`\sigma`.
        The new configurations :math:`\sigma'` should be a matrix with the same dimension as
        :math:`\sigma`.

        sampler: The Metropolis sampler.
        machine: A Flax module with the forward pass of the log-pdf.
        params: The PyTree of parameters of the model.
        sampler_state: The current state of the sampler. Should not modify it.
        key: A Jax PRNGKey to use to generate new random configurations.
        σ: The current configurations stored in a 2D matrix.

        returns : New configurations :math:`\sigma'` and None
        '''
        pass

