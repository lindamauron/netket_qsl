from typing import  Any
from netket.utils.types import NNInitFunc

import jax.numpy as jnp
import jax.nn.initializers as init
from jax import vmap

import flax.linen as nn
from ..lattice import Kagome as _Kagome


#######################################################################################################################
####################################################### Models ########################################################
#######################################################################################################################
def Psi_MF(phi, x):
    r'''
    Computes the mean-field part of an ansatz : log ψ^{MF}(σ_i) = sum_i log φ(σ_i)
    phi: array containing all phi coefficients (N,2)
    x : samples to evaluate (...,N)

    returns: log of mean-field probability (...)
    '''
    phi = jnp.array(phi)
    N = phi.shape[0]

    # compute the mean field part
    def one_mf(state):
        indices = jnp.array( (1+state)/2, dtype=int) # convert spins to indices -1,+1 -> 0,1
        
        # each sigma selects its phi
        def one_phi(i, index):
            return phi[i, index]
        
        # now for all components
        return vmap(one_phi)(jnp.arange(N), indices)
        
    # finally, differently for all samples
    return vmap(one_mf)(x)


class MF(nn.Module):
    r'''
    log \psi(z) = \sum_i \log( \phi(zi) )
    Notice that any one-body Jastrow term can be absorbed in the mean-field part, so we do not need it
    '''

    param_dtype : Any = jnp.complex128
    """The dtype of the weights."""

    mf_init : NNInitFunc = init.constant(1)
    """Initializer for the mean-field parameters."""

    @nn.compact
    def __call__(self, x):
        '''
        x : (Ns,N)
        '''
        N = x.shape[-1]

        phi = self.param(
            'ϕ', self.mf_init, (N,2), self.param_dtype
        )

        # # compute the mean field part
        # def one_mf(state):
        #     indices = jnp.array( (1+state)/2, dtype=int)
            
            
        #     def one_phi(i, index):
        #         return phi[i, index]
            
        #     return vmap(one_phi)(jnp.arange(N), indices)
        
        # mf = vmap(one_mf)(x)
        mf = Psi_MF(phi, x)
        
        return jnp.sum(jnp.log(mf), axis=-1)


class JMF_dense(nn.Module):
    r'''
    log \psi(z) = 1/2 \sum_ij z_i W_ij z_j + \sum_i \log( \phi(zi) )
    Notice that any one-body Jastrow term can be absorbed in the mean-field part, so we do not need it
    '''

    param_dtype : Any = jnp.complex128
    """The dtype of the weights."""

    jastrow_init : NNInitFunc = init.constant(0)
    """Initializer for the jastrow parameters."""
    
    mf_init : NNInitFunc = init.constant(1)
    """Initializer for the mean-field parameters."""

    @nn.compact
    def __call__(self, x):
        '''
        x : (Ns,N)
        '''
        N = x.shape[-1]
        
        #n = int(N*(N-1)/2)

        # Define the two variational parameters W and h
        W = self.param(
            'W', self.jastrow_init, (N,N), self.param_dtype
        )
        #W = jnp.zeros((N,N), dtype=self.param_dtype)
        #W = W.at[jnp.triu_indices(N,k=1)].set( v )
        #W = (W+W.T)/2 #symmetrize
        
        phi = self.param(
            'ϕ', self.mf_init, (N,2), self.param_dtype
        )
        
        
        # compute the nearest-neighbor correlations
        corr1=0.5*jnp.einsum( '...i,ij,...j',x,W,x )


        # # compute the mean field part
        # def one_mf(state):
        #     indices = jnp.array( (1+state)/2, dtype=int)
            
            
        #     def one_phi(i, index):
        #         return phi[i, index]
            
        #     return vmap(one_phi)(jnp.arange(N), indices)
        
        # mf = vmap(one_mf)(x)
        mf = Psi_MF(phi, x)
        
        return corr1+jnp.sum(jnp.log(mf), axis=-1)

class JMF_inv(nn.Module):
    r'''
    log \psi(z) = 1/2 \sum_ij z_i W_dij z_j + \sum_i \log( \phi(zi) )
    Notice that any one-body Jastrow term can be absorbed in the mean-field part, so we do not need it
    '''
    lattice : _Kagome
    """The lattice on which the model acts."""

    # n_neighbors : int
    """The number of neighboors possible on the lattice."""

    param_dtype : Any = jnp.complex128
    """The dtype of the weights."""

    jastrow_init : NNInitFunc = init.constant(0)
    """Initializer for the jastrow parameters."""
    
    mf_init : NNInitFunc = init.constant(1)
    """Initializer for the mean-field parameters."""
    


    @nn.compact
    def __call__(self, x):
        '''
        x : (Ns,N)
        '''
        N = x.shape[-1]

        # Define the two variational parameters W and h
        W = self.param(
            'W', self.jastrow_init, (self.lattice.n_distances,), self.param_dtype
        )
        
        phi = self.param(
            'ϕ', self.mf_init, (N,2), self.param_dtype
        )
        
        # compute the nearest-neighbor correlations
        corr = 0.5*jnp.einsum( '...i,ij,...j',x,W[self.lattice.neighbors_distances],x )

        # # compute the mean field part
        # def one_mf(state):
        #     indices = jnp.array( (1+state)/2, dtype=int)
            
            
        #     def one_phi(i, index):
        #         return phi[i, index]
            
        #     return vmap(one_phi)(jnp.arange(N), indices)
        
        # mf = vmap(one_mf)(x)
        mf = Psi_MF(jnp.array(phi), x)
        
        return corr + jnp.sum(jnp.log(mf), axis=-1)

class JMF3_inv(nn.Module):
    r'''
    log \psi(z) = \sum_ij z_i W_dij z_j + \sum_ijk W_dij_djk z_i z_j z_k + \sum_i \log( \phi(zi) )
    Notice that any one-body Jastrow term can be absorbed in the mean-field part, so we do not need it
    '''
    lattice : _Kagome
    """The lattice on which the model acts."""

    # n_neighbors : int
    """The number of neighboors possible on the lattice."""

    param_dtype : Any = jnp.complex128
    """The dtype of the weights."""

    jastrow_init : NNInitFunc = init.constant(0)
    """Initializer for the jastrow parameters."""
    
    mf_init : NNInitFunc = init.constant(1)
    """Initializer for the mean-field parameters."""
    


    @nn.compact
    def __call__(self, x):
        '''
        x : (Ns,N)
        '''
        N = x.shape[-1]

        # Define the two variational parameters W and h
        W2 = self.param(
            'W2', self.jastrow_init, (self.lattice.n_distances,), self.param_dtype
        )

        W3 = self.param(
            'W3', self.jastrow_init, (self.lattice.n_distances,self.lattice.n_distances), self.param_dtype
        )
        
        phi = self.param(
            'ϕ', self.mf_init, (N,2), self.param_dtype
        )
        
        # 2 bodies jastrow
        J2 = 0.5*jnp.einsum( '...i,ij,...j',x,W2[self.lattice.neighbors_distances],x )

        # 3 bodies jastrow
        J3 = jnp.einsum( 'ijm,ni->mjn',W3[self.lattice.neighbors_distances],x ) #sigma_i, sigma_j,n=Ns (n_neighbors,N,Ns)
        J3 = jnp.einsum( 'ijjn,ni,nj->n',J3[self.lattice.neighbors_distances],x,x ) #sigma_i, sigma_j n=Ns

        # # compute the mean field part
        # def one_mf(state):
        #     indices = jnp.array( (1+state)/2, dtype=int)
            
            
        #     def one_phi(i, index):
        #         return phi[i, index]
            
        #     return vmap(one_phi)(jnp.arange(N), indices)
        
        # mf = vmap(one_mf)(x)
        mf = Psi_MF(phi, x)
        
        return J2 + J3 + jnp.sum(jnp.log(mf), axis=-1)
    