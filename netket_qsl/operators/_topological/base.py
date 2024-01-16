from typing import Tuple
import abc
from functools import partial, lru_cache

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import scipy.sparse

from typing import Union, Optional

import netket as nk
from netket.hilbert import Spin as _SpinHilbert
from netket.operator._abstract_operator import AbstractOperator
from netket.utils.types import Array, ArrayLike, DType
import netket.jax as nkjax
from scipy.sparse import csr_matrix as _csr_matrix
from numbers import Number


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
                the operator is read in the matrix order, i.e. from right to left.
                This means that it is applied starting by sites[-1] and so on => the representation changes the order of the sites
        scalar : the constant multiplying the operator 
        '''
        
        sites = np.array(sites, int) # convert to int since we want to iterate over
        # to be able to iterate over even for a single site
        if sites.ndim == 0:
            sites = sites.reshape((1))
        if len(sites)==0:
            raise AttributeError(f"The operator needs at least one site to be declared, instead got {sites}")
        self._sites = sites

        self._scalar = scalar
            
        super().__init__(hilbert)
        

    @property
    def sites(self):
        return self._sites
    
    @property
    def scalar(self):
        return self._scalar
    
    @property
    def dtype(self) -> DType:
        '''
        The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩.
        '''
        return float
    
    def __getitem__(self, item):
        return self.sites[item]
    
    def draw(self):
        state = np.zeros(self.hilbert.size, dtype=int)
        state[self.sites] = 1
        
        colors = ['k', 'b']
        return state, colors
        
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


    def __mul__(self, other:Union["TopoOperator",Number]):
        '''
        Possibility to multiply two same operators OR the op by a number, nothing else
        Multiplying T1*T2 first applies T2 then T1 (for any TopoOperator T)
        '''
        # self*other
        if isinstance(other, type(self)):
            # we append the sites and cancel the ones touching since O*O = 1 for all O
            sitesA = self.sites
            sitesB = other.sites
            # while len(sitesA)>0 and len(sitesB)>0 and sitesA[-1] == sitesB[0]:
            #     sitesA = sitesA[:-1]
            #     sitesB = sitesB[1:]
            new_sites = np.append( sitesA, sitesB )
            
            return type(self)(self.hilbert, new_sites, self.scalar*other.scalar)
        
        elif np.issubdtype(type(other), np.number):
            return type(self)(self.hilbert, self.sites, self.scalar*other)
        
        return other.__rmul__(self)
        # except NotImplementedError:
        #     try:
        #         return Product(self.hilbert,[self,other], self.scalar)
            
        #     except AttributeError:
        #         raise NotImplementedError
        
        # raise NotImplementedError
            
        
    def __rmul__(self, other:Union["TopoOperator",Number]):
        # other * self
        if isinstance(other, type(self)):
            return other.__mul__(self)
        
        elif np.issubdtype(type(other), np.number):
            return type(self)(self.hilbert, self.sites, self.scalar*other)
        
        return Product(self.hilbert, [other,self], self.scalar)
                
    def __neg__(self) -> "TopoOperator":
        return -1.0*self
    
    def __rtruediv__(self,other):
        # other / self
        return other * self.T
    
    def __truediv__(self,other):
        #self/other
        return self * (1/other)
    
    
    @property
    def T(self):
        return type(self)(self.hilbert, self.sites[::-1], self.scalar)
        
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
    


from netket.jax import vmap_chunked
@partial(jit, static_argnames=("logpsi", "chunk_size"))
def o_loc(logpsi, pars, sigma, extra_args, chunk_size):
    '''
    Local energy of the operator O_loc(x) = sum_x' <x|O|x'> psi(x')/psi(x)

    returns : local value of the operator (Ns,)
    '''

    eta, mels = extra_args
    # check that sigma has been reshaped to 2D, eta is 2D
    # sigma is (Nsamples, Nsites)
    assert sigma.ndim == 2
    # eta is (Nsamples, Nsites) since there is only 1 connected element for each sample
    assert eta.ndim == 2
    
    # let's write the local energy assuming a single sample, and vmap it
    @partial(vmap_chunked, in_axes=(0, 0, 0), chunk_size=chunk_size)
    def _loc_vals(sigma, eta, mels):
        return mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma))
    
    return _loc_vals(sigma, eta, mels)


# @nk.vqs.get_local_kernel.dispatch
# def get_local_kernel(vstate: nk.vqs.MCState, op: TopoOperator):
#     return o_loc

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: TopoOperator, chunk_size: Optional[int]):
    '''
    Compute the extra arguments for the o_loc operators, i.e. the connected elements and matrix elements

    for samples of shape (n_chains,n_samples,N) 
    returns: Tuple (eta,mels) containing the connected elements (eta) and matrix elements (mel) #(n_chains,n_samples,N), (n_chains_n_samples,)
    '''
    sigma = vstate.samples
    
    # get the connected elements. 
    extra_args = vmap_chunked(op.get_conns_and_mels, chunk_size=chunk_size)(sigma) #(n_chains,n_samples,N), (n_chains_n_samples,)
    return sigma, extra_args

@lru_cache(5)
def sparsify(Ô:TopoOperator):
    """
    Converts to sparse but also cache the sparsificated result to speed up.
    """
    return Ô.to_sparse()


class Product(TopoOperator):
    '''
    Instance to use when multiplying topological operators:
    Multiplication(hi,[P,Q]).apply(psi) = Q@P|psi>
    This takes into account the fact that operators of the same type commute, while not doing any assumption on other relations.
    
    This class then allows for efficient expectation values of multiplied topological operators on large scales of the lattice.
    '''

    def __init__(self,hilbert,operators,scalar=1.0):
        for op in operators:
            if not np.issubdtype(type(op), TopoOperator):
                raise AttributeError(f"Product works only on operators, instead got {op}")
            
            if hilbert != op.hilbert:
                raise ValueError(f'All hilbert spaces must be identical, got {hilbert} and {op.hilbert}')
        self._hilbert = hilbert

        # assembles the operators by appending sites if same type (can reduce the complexity of the operator)
        ops = []
        i = 0
        while i < len(operators):
            k=i+1
            op = operators[i]

            while k<len(operators) and type(operators[i]) == type(operators[k]):
                op = op*operators[k]
                operators = np.delete(operators, k)
            else:
                i += 1
            ops.append( op )
            
        # Now that we simplified averything we could, make a proper list
        operators = []
        for op in ops:
            scalar *= op.scalar
            operators.append(op*(1/op.scalar))

        self._operators = np.array(operators)
        self._scalar = scalar

        self.n_operators = len(self.operators)

    @property
    def operators(self):
        return self._operators
    
    def draw(self):
        assert len(self.operators) <= 10, 'The drawing for so many operators is not defined'

        colors = ['k']
        state = np.zeros(self.hilbert.size, dtype=int)
        for k,op in enumerate(self.operators):
            state[op.sites] = k+1
            colors.append(f'C{k}')
    
        return state, colors
    

    def __getitem__(self, item):
        return self.operators[item]
    

    def _conn_one_triangle(self, x: Array, i: int) -> Tuple[Array, Array]:
        m = 1

        for op in self.operators:
            if i in op.sites:
                x, new_m = op._conn_one_triangle(x, i)
            m = new_m * m

        return x, m
    
    def get_conns_and_mels(self, sigma):
        ms = jnp.ones(sigma.shape[0])

        for i in range(self.n_operators):
            sigma, new_ms = self[i].get_conns_and_mels(sigma)
            ms *= new_ms
        return sigma, self.scalar*ms


    def __repr__(self):
        str = f'{self.scalar}*Product(hilbert={self.hilbert}, '
        for i in range(self.n_operators):
            if i!=0:
                str += '@'
            str += f" {type(self[i]).__name__}({self[i].sites}) "
        return str+f', dtype={self.dtype})'

    def __mul__(self, other):
        '''
        '''
        #self*other
        if isinstance(other, Product):
            ops = []
            for o in self.operators:
                ops.append(o)
            for o in other.operators:
                ops.append(o)
            return Product(self.hilbert, ops, self.scalar*other.scalar)
        
        elif isinstance(other, TopoOperator):
            # i.e. self * O
            ops = list(self.operators)
            ops.append(other)
            return Product(self.hilbert,ops, scalar=self.scalar)
        
        elif np.issubdtype(type(other), np.number):
            return type(self)(self.hilbert, self.operators, self.scalar*other)

        raise NotImplementedError
            
    def __rmul__(self,other):   
        #other * self  
        if isinstance(other, TopoOperator):
            ops = list(self.operators)[::-1]
            ops.append(other)
            return Product(self.hilbert,ops[::-1], scalar=self.scalar)
        elif np.issubdtype(type(other), np.number):
            return type(self)(self.hilbert, self.operators, self.scalar*other)
        
        return NotImplementedError


    
    @property
    def T(self):
        ops = []
        for o in self.operators:
            ops.append(o.T)
        return Product(self.hilbert, ops[::-1], self.scalar)

    def to_sparse(self):
        '''
        Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        return : sparse matrix representation of the operator.
        '''
        res = scipy.sparse.eye(self.hilbert.n_states)
        for op in self.operators:
            res = res@( op.to_sparse() )

        return res