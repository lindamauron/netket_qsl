from typing import Tuple, Optional

import netket as nk
from netket.utils.types import PyTree, PRNGKeyT
from netket import sampler

import jax 
import jax.numpy as jnp
from jax import jit, vmap
from flax import linen as nn

from ...lattice import neighbors
from .._restricted import RestrictedRule


@jit
def _flip_one_swap_(s,index):
    '''
    Allows to flip one spin while maintaining the condition of only one excited state by triangle, swapping j,k
    s : single initial state
    index : spin to flip
    
    return : spin chain with spins accordingly flipped (N,)
    '''

    j,k = neighbors(index) # neighbors in the triangle

    #|ggg> <-> |rgg>, |grg>,|ggr>->|rgg> <=> i -> -i & j,k -> -1
    # i.e i is always flipped and j,k always end up in the gs
    s = s.at[index].set( -s[index] )
    s = s.at[j].set( -1 )
    s = s.at[k].set( -1 )

    return s

@nk.utils.struct.dataclass
class TriangleRuleSwap(RestrictedRule):
    r'''
    A transition rule acting on one triangle of the lattice.
    Can be seen as choosing one triangle of the lattice which Hilbert's space is \{ |000>, |100>, |010>, |001> \}.
    
    The transition probability is in two steps : 
    1. A site i is randomly chosen with uniform probability
    2. From this site, the site is flipped, but the same-triangle-neighbors are also impacted in order to stay in the restricted space. 
       Therefore, this rule can result in one or two spin flips depending on the initial situations. 
       Say i=0, the transition is : 
       |ggg> -> |rgg>
       |rgg> -> |ggg>
       |grg> -> |rgg>
       |ggr> -> |rgg>
    '''
    
    def transition(self, sampler: "sampler.MetropolisSampler", machine: nn.Module, params: PyTree, sampler_state: "sampler.SamplerState", key: PRNGKeyT, σ: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        r'''
        It proposes a new configuration set of configurations $\sigma'$ starting from the current
        chain configurations :math:`\sigma`.

        sampler: The Metropolis sampler.
        machine: A Flax module with the forward pass of the log-pdf.
        params: The PyTree of parameters of the model.
        sampler_state: The current state of the sampler. Should not modify it.
        key: A Jax PRNGKey to use to generate new random configurations.
        σ: The current configurations stored in a 2D matrix.

        returns : New configurations :math:`\sigma'` and None
        '''
        
        n_chains = σ.shape[0]
        N = int(sampler.hilbert.size)

        # split the random key
        key_idx, _ = jax.random.split(key,2)
        
        # select what spins to flip (1 per chain)
        indices = jax.random.randint(key_idx, minval=0, maxval=N, shape=(n_chains,1))

        # flip all chains at once
        σ = vmap(_flip_one_swap_, in_axes=(0,0), out_axes=0)(σ, indices)
        
        return σ, None

    def __repr__(self):
        '''
        Representation of the class
        '''
        return f'TriangleRule_swaped()'

