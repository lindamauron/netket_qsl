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
def _flip_one_Q_(s, index):
    '''
    Applies the Q operator at spin index on one sample 
    s : initial state
    index : spin to flip
    
    return : spin chain with spins accordingly flipped (N,)
    '''
    j,k = neighbors(index)

    # |ggg> <-> |rgg> & |grg> <-> |ggr>
    # i.e. we always swap j<->k and flip i if j*k=1
    s_prime = s.at[index].set( -s[index]*s[j]*s[k] )
    s_prime = s_prime.at[j].set( s[k] )
    s_prime = s_prime.at[k].set( s[j] )

    return s_prime

@nk.utils.struct.dataclass
class TriangleRuleQ(RestrictedRule):
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
       |grg> -> |ggr>
       |ggr> -> |grg>

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
        
        #number of chains
        n_chains = σ.shape[0]
        N = σ.shape[-1]

        # split the random key
        key, _ = jax.random.split(key)

        # chose on which site to act
        sites = jax.random.choice(key, N, shape=(n_chains,) )

        return vmap(_flip_one_Q_, in_axes=(0, 0), out_axes=0 )(σ, sites), None
    
    def __repr__(self):
        '''
        Representation of the class
        '''
        return f'TriangleRule_Q()'

