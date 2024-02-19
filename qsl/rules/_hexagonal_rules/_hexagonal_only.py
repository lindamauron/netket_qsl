import numpy as np
from typing import Tuple, Optional
import warnings
from textwrap import dedent

import netket as nk
from netket.utils.types import PyTree, PRNGKeyT
from netket import sampler

import jax 
import jax.numpy as jnp
from jax import vmap
from flax import linen as nn

from ...lattice._kagome import Kagome as _Kagome
from .._restricted import RestrictedRule

from .base import _global_transition_batch


@nk.utils.struct.dataclass
class HexagonalRule(nk.sampler.rules.MetropolisRule):
    r"""
    Global rule for which one randomly chooses an hexagon of the lattice and applies the Q operator. 

    Construction : 
    lattice: lattice on which the state acts
    hexs: filled hexagons of the lattice (jax-compatible)
    """
    hexs : jnp.ndarray
    def __pre_init__(self,lattice:_Kagome,*args,**kwargs):
        """
        Prepares the class attribute hexs
        """
        warnings.warn(dedent('This sampling rule is highly not ergodic (except for thermalized RVB states).'),
                      category=DeprecationWarning
                      )
        kwargs['hexs'] = jnp.array(lattice.hexagons.filled)
        return args, kwargs
    

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
        # #number of chains
        # n_chains = σ.shape[0]
        # # number of hexagons
        # n = self.hexs.shape[0]

        # # split the random key
        # key, _ = jax.random.split(key,2)


        # # choose for each hexagon wether or not we apply the string
        # conds = jax.random.choice(key, np.array([True,False]), shape=(n_chains,n), p=jnp.array([self.p, 1-self.p]) )
        
        
        # def _one_transition(x, conds):
            
        #     def body_fun(i,x):
        #         return jax.lax.cond( conds[i], _apply, _do_nothing, x, self.hexs[i] )
        
        #     x = jax.lax.fori_loop(0, n, body_fun, x)

        #     return x

        # return vmap(_one_transition, in_axes=(0,0) )(σ, conds), None
        return _global_transition_batch(key,σ,self.hexs), None

    def __repr__(self):
        '''
        Representation of the class
        '''
        return f'HexagonalRule(# of hexagons: {self.hexs.shape[0]})'


@nk.utils.struct.dataclass
class RestrictedHexagonalRule(RestrictedRule):
    r'''
    A transition rule acting on the hexagons of the lattice.
    Can be seen as applying Q on all sites of one random hexagons, where the Hilbert space is \{ |000>, |100>, |010>, |001> \}^N/3.
    When applied on a dimerization, this transition should result in another correct dimerization.
       
    Construction:
    lattice: lattice on which the state acts
    hexs: filled hexagons of the lattice (jax-compatible)
    '''
    hexs : jnp.ndarray
    def __pre_init__(self,lattice:_Kagome,*args,**kwargs):
        """
        Prepares the class attribute hexs
        """
        warnings.warn(dedent('This sampling rule is highly not ergodic (except for thermalized RVB states).'),
                      category=DeprecationWarning
                      )
        kwargs['hexs'] = jnp.array(lattice.hexagons.filled)
        return args, kwargs


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
        # #number of chains
        # n_chains = σ.shape[0]
        # # number of hexagons
        # n = self.hexs.shape[0]

        # # split the random key
        # key, _ = jax.random.split(key,2)


        # # choose for each hexagon wether or not we apply the string
        # conds = jax.random.choice(key, np.array([True,False]), shape=(n_chains,n), p=jnp.array([self.p, 1-self.p]) )
        
        
        # def _one_transition(x, conds):
            
        #     def body_fun(i,x):
        #         return jax.lax.cond( conds[i], _apply, _do_nothing, x, self.hexs[i] )
        
        #     x = jax.lax.fori_loop(0, n, body_fun, x)

        #     return x

        # return vmap(_one_transition, in_axes=(0,0) )(σ, conds), None
        return _global_transition_batch(key,σ,self.hexs), None


    def __repr__(self):
        '''
        Representation of the class
        '''
        return f'RestrictedHexagonalRule(# of hexagons: {self.hexs.shape[0]})'

