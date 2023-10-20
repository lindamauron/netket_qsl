from typing import Tuple, Optional

import netket as nk
from netket.utils.types import PyTree, PRNGKeyT
from netket import sampler

import jax 
import jax.numpy as jnp
from jax import vmap, jit
from flax import linen as nn

from .base import _global_transition



@jit
def _local_transition(key, σ, hexs):
    return _local_transition_batch(key, σ.reshape(1,-1), hexs).reshape(-1)

@jit
def _local_transition_batch(key, σ, hexs):
    key1, key2 = jax.random.split(key, 2)

    n_chains = σ.shape[0]
    N = σ.shape[-1]

    indxs = jax.random.randint(key1, shape=(n_chains,), minval=0, maxval=N)

    def _flip(s, i):
        return s.at[i].set( -s[i] )

    σp = vmap(_flip, in_axes=(0,0))(σ, indxs)

    return σp



@nk.utils.struct.dataclass
class MixedRule(nk.sampler.rules.MetropolisRule):
    '''
    Transition rule that mixes local moves (single spin flip) and global moves (Q operator on a hexagon)
    The process is ; 
    1. The type of move is chosen (global with probability p_global)
    2. The move is applied
        2a. For global, Q is applied on each hexagon with a probability mean/n_hexagons, so that the mean number of applied hexagons corresponds to mean_global
        2b. For local, a single spin flip is applied on one of the sites
        
    
    hexs : container of the hexagons on the lattice
    p_global : the probability to do a global move (then 1-p_global is the prob to do a local move)
    mean_global : the mean number of hexagons we want to be switched for a global move
                    notice that mean=3 ~ mean=1 and mean=4~mean=0
    '''
    hexs : jnp.ndarray

    p_global : float = 0.5
    
    def __repr__(self):
        '''
        Representation of the class
        '''
        if self.hexs.shape[0] <= 6:
            return f'MixedRule( p_global = {self.p_global}, {self.hexs.shape[0]} hexagons: {self.hexs})'

        return f'MixedRule( p_global = {self.p_global}, {self.hexs.shape[0]} hexagons)'

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

        # split the random key
        key, _ = jax.random.split(key)
        
        conds = jax.random.choice(key, jnp.array([True,False]), shape=(n_chains,), p=jnp.array([self.p_global,1-self.p_global]) )
        keys = jax.random.split(key,n_chains)

        def _one_chain(key, condition, s):
            return jax.lax.cond(condition, _global_transition, _local_transition, key, s, self.hexs)
        
        return vmap(_one_chain, in_axes=(0, 0, 0), out_axes=0 )(keys, conds, σ), None
