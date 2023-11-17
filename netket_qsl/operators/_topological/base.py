from typing import Tuple
import abc
from functools import partial

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import scipy.sparse

from typing import Union

import netket as nk
from netket.hilbert import Spin as _SpinHilbert
from netket.operator._abstract_operator import AbstractOperator
from netket.utils.types import Array, ArrayLike, Dtype
from scipy.sparse import csr_matrix as _csr_matrix
""" 
Was tested and works
Is it better to use than the previous operator ???
Says so in NetKet's API
"""

class TopoOperator(AbstractOperator):
    '''
    Super class for the Topological operators. 
    This is defined from the beginning since it will be applied on many sites and is by definition not a local operator. 
    This implementation allows to travel on many sites without crashing since the computation is done one triangle at a time. 

    It is possible to multiply multiple topological operators of the same type (type P,Q,R) or to multiply them by a scalar. No other operation. 
    '''
    def __init__(self, hilbert:_SpinHilbert, sites:Union[int,Array,ArrayLike], scalar=1.0):
        '''
        Instantiates one topological operator 
        hilbert : Hilbert space on which the operator acts
        sites : the sites of the lattice where we want to apply the operator
                can be a single site i or a list of them [i,j,...]
                the operator is then applied starting by sites[0] and so on
        scalar : the constant multiplying the operator 
        '''
        
        self.sites = np.array(sites, int) # convert to int since we want to iterate over
        # to be able to iterate over even for a single site
        if self.sites.ndim == 0:
            self.sites = self.sites.reshape((1))

        self.scalar = scalar
            
        super().__init__(hilbert)
        

    @property
    def dtype(self) -> Dtype:
        '''
        The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩.
        '''
        return float
        
        
    @property
    def max_conn_size(self) -> int:
        '''
        The maximum number of non zero ⟨x|O|x'⟩ for every x.
        In the case of the Topological operators, it is always 1
        '''
        return 1

    @abc.abstractmethod
    def _conn_one_triangle(self, x:jnp.ndarray, i:int) -> Tuple[jnp.ndarray, jnp.ndarray] :
        '''
        The conn and m_el of the operator applied on one site (i)
        Has to be defined in the class for it to be instantiable, should be jitted for the operator to be faster
        
        x : flat sample (Ns,N)
        i : single site on which the operator is acting

        return : σ (Ns,N), <x|O|σ> (Ns,)
        '''

    @partial(jit, static_argnums=0)
    def get_conns_and_mels(self, sigma:Array):
        '''
        Gives the connections x' and the corresponding matrix elements m=<x|O|x'> of the operator
        sigma : array of states for which one needs the connections and matrix elements (Ns,N)

        returns : conns (Ns,N) and m_els (Ns,)
        '''
        # loop over the sites, x as input and output, m_primes from each loop
        x_primes, m_primes = jax.lax.scan(self._conn_one_triangle, sigma, self.sites)

        # in the end, we return the final x and multiply all the m_els
        return x_primes, self.scalar*jnp.prod(m_primes, axis=0)


    def __mul__(self, other:Union["TopoOperator",np._NumberType]):
        '''
        Possibility to multiply two same operators OR the op by a number, nothing else
        Multiplying T1*T2 first applies T2 then T1 (for any TopoOperator T)
        '''
        if isinstance(other, type(self)):
            
            # we append the sites so that we first apply the string the most on the right (since they do not commute)
            new_sites = np.append( other.sites, self.sites)
            
            return type(self)(self.hilbert, new_sites, self.scalar)
        
        if not np.issubdtype(type(other), np.number):
            raise NotImplementedError
            
        return type(self)(self.hilbert, self.sites, self.scalar*other)

        
    def __rmul__(self, other:Union["TopoOperator",np._NumberType]):
        if isinstance(other, type(self)):
            return other.__mul__(self)
        
        if not np.issubdtype(type(other), np.number):
            raise NotImplementedError
            
        return type(self)(self.hilbert, self.sites, self.scalar*other)
    
    def __neg__(self) -> "TopoOperator":
        return -1.0*self
        
    def __repr__(self):
        return f"{self.scalar}*{type(self).__name__}(hilbert={self.hilbert}, sites={self.sites}, dtype={self.dtype})"
    
    def to_sparse(self) -> _csr_matrix:
        '''
        Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        return : sparse matrix representation of the operator.
        '''

        basis = self.hilbert.all_states()

        #sections = np.empty(basis.shape[0], dtype=np.int32)
        x_prime, mels = self.get_conns_and_mels(basis)

        numbers = self.hilbert.states_to_numbers(x_prime)

        sections1=np.arange(self.hilbert.n_states+1, dtype=np.int32)

        return scipy.sparse.csr_matrix(
            (mels, numbers, sections1),
            shape=(self.hilbert.n_states, self.hilbert.n_states),
        )

    def to_dense(self) -> Array:
        '''
        Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        return : dense matrix representation of the operator as a Numpy array.
        '''
        return self.to_sparse().todense()
    




def o_loc(logpsi, pars, sigma, extra_args):
    '''
    Local energy of the operator O_loc(x) = sum_x' <x|O|x'> psi(x')/psi(x)

    returns : local value of the operator (Ns,)
    '''

    eta, mels = extra_args
    # check that sigma has been reshaped to 2D, eta is 3D
    # sigma is (Nsamples, Nsites)
    assert sigma.ndim == 2
    # eta is (Nsamples, Nconnected, Nsites)
    assert eta.ndim == 2
    
    # let's write the local energy assuming a single sample, and vmap it
    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma))
    
    return _loc_vals(sigma, eta, mels)

@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: TopoOperator):
    return o_loc

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: TopoOperator):
    '''
    Compute the extra arguments for the o_loc operators, i.e. the connected elements and matrix elements
    '''
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = op.get_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size))
    return sigma, extra_args


@nk.vqs.expect.dispatch
def expect(vstate: nk.vqs.MCState, op:TopoOperator):
    sigma, extra_args = get_local_kernel_arguments(vstate, op)

    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    sigma = sigma.reshape(-1, N)

    #eta, mels = extra_args


    E_loc = o_loc(vstate._apply_fun, vstate.variables, sigma, extra_args).reshape(-1, n_chains)


    return nk.stats.statistics(E_loc)



from functools import  lru_cache
from netket.stats import Stats


@lru_cache(5)
def sparsify(Ô):
    """
    Converts to sparse but also cache the sparsificated result to speed up.
    """
    return Ô.to_sparse()


@nk.vqs.expect.dispatch
def expect(vstate: nk.vqs.FullSumState, Ô: TopoOperator) -> Stats:

    O = sparsify(Ô)
    Ψ = vstate.to_array()

    # TODO: This performs the full computation on all MPI ranks.
    # It would be great if we could split the computation among ranks.

    OΨ = O @ Ψ
    expval_O = (Ψ.conj() * OΨ).sum()

    variance = jnp.sum(jnp.abs(OΨ - expval_O * Ψ) ** 2)
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)

