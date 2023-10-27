import numpy as np

from typing import Any
from netket.utils.types import NNInitFunc

import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn


class Rotation(nn.Module):
    r"""
    Defines an ansatz $ \psi(Z) = J(Z) <Z|Rz(\gamma) Ry(\beta) Rx(\alpha) |0> $
    
    For now, the angle \beta is shifted so that for alpha,beta,gamma=0, psi(z) = 1 for all z (|psi> = |+>)
    """
    
    angles_init : NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the angles."""
    
    angles_dtype : Any = np.float64
    """The dtype of the angles."""
 
    @nn.compact
    def __call__(self, x):
        '''
        x : (...,N)
        '''
        N = x.shape[-1]

        # Define the variational parameters
        alpha = self.param(
            "α", self.angles_init, (N,), self.angles_dtype
        )
        beta = self.param(
            "β", self.angles_init, (N,), self.angles_dtype
        )
        gamma = self.param(
            "γ", self.angles_init, (N,), self.angles_dtype
        )
        

        ## defines the <x| Ry Rx |0> (since Rz is diagonal)
        #RyRx = jnp.cos(alpha/2)*jnp.cos(beta/2-np.pi/4-x*np.pi/4) - 1j*x*jnp.sin(alpha/2)*jnp.sin(beta/2+np.pi/4+x*np.pi/4)

        # Replace beta by beta+pi/2 to start on the |+> state if all parameters are set to zero
        RyRx = jnp.cos(alpha/2)*jnp.cos(beta/2-x*np.pi/4) - 1j*x*jnp.sin(alpha/2)*jnp.sin(beta/2+np.pi/2+x*np.pi/4)
        
        
        return jnp.sum(1j*gamma/2*x + jnp.log(RyRx), axis=-1)


class RotJ(nn.Module):
    r"""
    Defines an ansatz $ \psi(Z) = J(Z) <Z|Rz(\gamma) Ry(\beta) Rx(\alpha) |0> $
    if use_jastrow is set to false, the ansatz becomes a meanfield one (but more meaningfull)

    Carefull : for now, the angles and kernel must share their dtype, so do what is more meaningfull
    """
    
    angles_init : NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the angles."""
    
    kernel_init : NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the jastrow kernel."""
    
    kernel_dtype : Any = complex
    """The dtype of the jastrow kernel."""

    @nn.compact
    def __call__(self, x):
        '''
        x : (...,N)
        '''
        N = x.shape[-1]

        # Define the variational parameters
        alpha = self.param(
            "α", self.angles_init, (N,), np.float64
        )
        beta = self.param(
            "β", self.angles_init, (N,), np.float64
        )
        gamma = self.param(
            "γ", self.angles_init, (N,), np.float64
        )
        

        ## defines the <x| Ry Rx |0> (since Rz is diagonal)
        #RyRx = jnp.cos(alpha/2)*jnp.cos(beta/2-np.pi/4-x*np.pi/4) - 1j*x*jnp.sin(alpha/2)*jnp.sin(beta/2+np.pi/4+x*np.pi/4)
        RyRx = jnp.cos(alpha/2)*jnp.cos(beta/2-x*np.pi/4) - 1j*x*jnp.sin(alpha/2)*jnp.sin(beta/2+np.pi/2+x*np.pi/4)
        
        W_RE = self.param(
            "W_RE", self.kernel_init, (N, N), np.float64
        )

        # Do we put the complex jastrow term or not
        if self.kernel_dtype == complex : 
            W_IM = self.param(
                "W_IM", self.kernel_init, (N, N), np.float64
            )
            W = W_RE + 1j*W_IM
        else :
            W = W_RE

        W = W + W.T
        
        return jnp.sum(1j*gamma/2*x + jnp.log(RyRx), axis=-1) + jnp.einsum('...i,ij,...j', x, W, x)
    