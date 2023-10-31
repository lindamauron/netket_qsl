from typing import Tuple, Optional

import netket as nk
from netket.utils.types import PyTree, PRNGKeyT
from netket import sampler

import jax 
import jax.numpy as jnp
from jax import vmap, jit
from flax import linen as nn

from .base import _global_transition
from .._restricted import RestrictedRule
from .._triangle._Q import _flip_one_Q_


@jit
def _restr_local_transition(key, σ, hexs):
    return _restr_local_transition_batch(key, σ.reshape(1,-1), hexs).reshape(-1)

@jit # CHANGE HERE FOR DIFFERENT LOCAL RULES (_flip_one_swap_ or _flip_one_unchanged_ or _flip_one_Q_)
def _restr_local_transition_batch(key, σ, hexs):
    '''
    applies the local transition on the restricted space
    key: A Jax PRNGKey to use to generate new random configurations.
    σ: The current configurations stored in a 2D matrix.
    hexs: container of the haxagons on the lattice

    return: new configurations σ' (Ns,N)
    '''
    
    n_chains = σ.shape[0]
    N = σ.shape[-1]

    # split the random key
    key_idx, _ = jax.random.split(key,2)
    
    # select what spins to flip (1 per chain)
    indices = jax.random.randint(key_idx, minval=0, maxval=N, shape=(n_chains,1))

    # flip all chains at once
    σ = vmap(_flip_one_Q_, in_axes=(0,0), out_axes=0)(σ, indices)
    
    return σ


@nk.utils.struct.dataclass
class RestrictedMixedRule(RestrictedRule):
    '''
    Transition rule that mixes local moves (triangular spin flip) and global moves (Q operator on an hexagon) on the restricted space with at most one excitation per triangle
    
    So, we have to redefine the random_state, to start in the right space

    The process is :
    1. The type of move is chosen (global with probability p_global)
    2. The move is applied
        2a. For global, Q is applied on each hexagon with a probability of 1/2, so that the mean number of applied hexagons corresponds to n_hexagons/2 
            (this minimizes the probability to apply on zero/all hexagon)
        2b. For local, a spin is chosen to be flipped. There, one can choose the way to do this respecting staying in the restricted space. 
            Choice between : 
            - swap : if another spin was excited, it is is flipped and spin i is excited
            - unchanged : if another spin was excited, it stays this way and i is not flipped
            - Q : if another spin was excited, it is is swaped with the third spin and i is unchanged, as for the operator Q    
        
    hexs : container of the hexagons on the lattice
    p_global : the probability to do a global move (then 1-p_global is the prob to do a local move)
    '''
    hexs : jnp.ndarray
    p_global : float = 0.5

    def __repr__(self):
        '''
        Representation of the class
        '''
        if self.hexs.shape[0] <= 6:
            return f'RestrictedMixedRule( local=TriangleQ, p_global = {self.p_global}, {self.hexs.shape[0]} hexagons: {self.hexs})'

        return f'RestrictedMixedRule( local=TriangleQ, p_global = {self.p_global}, {self.hexs.shape[0]} hexagons)'

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
        
        # do we apply the global move (otherwise, local)
        conds = jax.random.choice(key, jnp.array([True,False]), shape=(n_chains,), p=jnp.array([self.p_global,1-self.p_global]) )
        keys = jax.random.split(key,n_chains)

        # applies the rule on one sample
        def _one_chain(key, condition, s):
            '''
            if condition:
                _global_transition(key, s, self.hexs)
            else: 
                _local_transition(key, s, self.hexs)
            '''
            return jax.lax.cond(condition, _global_transition, _restr_local_transition, key, s, self.hexs)
        
        # vmap for all batches
        return vmap(_one_chain, in_axes=(0, 0, 0), out_axes=0 )(keys, conds, σ), None
    


