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


from netket.utils import struct
@struct.dataclass
class MixedRuleState:
    probs: jnp.ndarray


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
    1. The type of move is chosen (sampler_state.rule_state.probs=[p_global,p_local])
    2. The move is applied
        2a. For global, Q is applied on one random hexagon
        2b. For local, Q is applied on one random site

    Construction : 
    lattice: lattice on which the state acts
    hexs: filled hexagons of the lattice (jax-compatible)
    initial_probs: probailities for the [0]:global move [1]local move
    '''
    hexs : jnp.ndarray
    initial_probs: jnp.ndarray

    def __pre_init__(self,lattice,probs,*args,**kwargs):
        """
        Prepares the class attribute hexs and probs
        """
        kwargs['hexs'] = jnp.array(lattice.hexagons.filled)
        ps = jnp.asarray(probs)

        if len(ps) != 2:
            raise ValueError(
                "Length mismatch between the probabilities and the rules: probabilities "
                f"has length {len(ps)} , rules has length 2."
            )
        
        if not jnp.allclose(jnp.sum(ps), 1.0):
            raise ValueError(
                "The probabilities must sum to 1, but they sum to "
                f"{jnp.sum(ps)}."
            )
        
        kwargs['initial_probs'] = ps/ps.sum()
        return args, kwargs

    def init_state(self, sampler, machine, params, key):
        return MixedRuleState(probs=self.initial_probs)
    

    def __repr__(self):
        '''
        Representation of the class
        '''
        if self.hexs.shape[0] <= 6:
            return f'RestrictedMixedRule( local=TriangleQ, p_initial = {self.initial_probs}, {self.hexs.shape[0]} hexagons: {self.hexs})'

        return f'RestrictedMixedRule( local=TriangleQ, p_initial = {self.initial_probs}, {self.hexs.shape[0]} hexagons)'

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
        conds = jax.random.choice(key, jnp.array([True,False]), shape=(n_chains,), p=sampler_state.rule_state.probs )
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
    


