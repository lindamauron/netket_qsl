import numpy as np 
from netket.operator.spin import sigmax,sigmaz, sigmay, sigmam, sigmap
from ._topological import P,Q,R

from ..hilbert import TriangleHilbertSpace as _TriangleHilbertSpace
from netket.operator._local_operator import LocalOperator as _LocalOperator
from netket.utils.types import DType as _DType

import jax.numpy as jnp
from functools import partial
from jax import jit


""" 
Defines all the operators needed in the Z basis, so that there is no need to define them in code
In practice, only needs to redefine X,Z,P,Q,R but the rest is changed in consequence

In general, call : 
>>> from ZOperators import r_density, TopoOps, delta_Hamiltonian
>>> n_op = r_density(hi, lattice)
>>> P_op, Q_op, R_op = TopoOps(hi, lattice, hex=0)
>>> H = delta_Hamiltonian(hi, lattice) # for H(Δ)
>>> H_dyn = time_Hamiltonian(hi, lattice) # for H(t)
"""


#######################################################################################################################
################################################## General Operators ##################################################
#######################################################################################################################
def restricted_sigmax(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = float
) -> _LocalOperator:
    '''
    Builds the :math:`σ^x` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''

    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^x)
    t = np.floor(i/3).astype(int) # the triangle index
    j = 3*t + (i+1)%3
    k = 3*t + (i+2)%3
    
    if i%3==0:
        index = 4
    elif i%3==1:
        index = 2
    else:
        index = 1
    
    # The matrix representing σ^x in the retricted basis (put coefficients outside space to 0)
    D = np.zeros((8,8), dtype=dtype)
    D[0,index] = 1.0
    D[index,0] = 1.0
    
    
    return _LocalOperator(hilbert, D, (i,j,k), dtype=dtype)

def restricted_sigmay(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = complex
) -> _LocalOperator:
    '''
    Builds the :math:`σ^y` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''

    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^x)
    t = np.floor(i/3).astype(int) # the triangle index
    j = 3*t + (i+1)%3
    k = 3*t + (i+2)%3
    
    if i%3==0:
        index = 4
    elif i%3==1:
        index = 2
    else:
        index = 1
    
    # The matrix representing σ^x in the retricted basis (put coefficients outside space to 0)
    D = np.zeros((8,8), dtype=dtype)
    D[0,index] = -1j
    D[index,0] = 1j
    
    
    return _LocalOperator(hilbert, D, (i,j,k), dtype=dtype)


def X(hilbert, i, restricted=True):
    '''
    sigma_x operator on site i
    hilbert : hilbert space of the system
    i : site on which to apply the operator
    
    returns : LocalOperator
    '''
    if isinstance(hilbert, _TriangleHilbertSpace) or restricted:
        return restricted_sigmax(hilbert,i)
    else:
        return sigmax(hilbert,i)

def Z(hilbert,i):
    '''
    sigma_z operator on site i
    hilbert : hilbert space of the system
    i : site on which to apply the operator
    
    returns : LocalOperator
    '''

    # since it is diagonal, we do not need to define one in the restricted space
    return sigmaz(hilbert,i)

def Y(hilbert,i,restricted=True):
    '''
    sigma_y operator on site i
    hilbert : hilbert space of the system
    i : site on which to apply the operator
    
    returns : LocalOperator
    '''
    if isinstance(hilbert, _TriangleHilbertSpace) or restricted:
        return restricted_sigmay(hilbert,i)
    else:
        return sigmay(hilbert,i)

def g_occ(hilbert,i) : 
    '''
    g projector = |g><g| on site i
    hilbert : hilbert space of the system
    i : site on which to apply the operator
    
    returns : LocalOperator
    '''
    return sigmap(hilbert,i)*sigmam(hilbert,i)

def r_occ(hilbert,i) : 
    '''
    r projector = |r><r| on site i = n_i (defined this way to avoid numerous computations)
    hilbert : hilbert space of the system
    i : site on which to apply the operator
    
    returns : LocalOperator
    '''
    return sigmam(hilbert,i)*sigmap(hilbert,i)


def r_density(hilbert, lattice):
    '''
    mean of the r_occ on the whole lattice 1/N \sum_i n_i
    hilbert : hilbert space of the system
    lattice : lattice on which we want the mean (gives the number of sites)

    returns : LocalOperator
    '''
    N = lattice.N

    # The total number of Rydberg excitations on the lattice
    N_op = sum([r_occ(hilbert,i) for i in range(N)])

    # Mean number of Rydberg exitations
    return N_op/N

def TopoOps(hilbert, lattice, hex=0, sites=None):
    '''
    Constructs the topological operators on a lattice
    one can either apply on a specifi hexagon or on a (list of) site(s)
    default : acting on hexagon 0

    hilbert : hilbert space of the system
    lattice : lattice on which the operators act
    hex : hexagon around which the operators should apply
    sites : list of sites on which the operators should apply

    returns : callable operators P,Q,R
    '''
    if sites is None:
        sites = lattice.hexagons.bulk[hex]


    return P(hilbert, sites), Q(hilbert, sites), R(hilbert, sites)


@partial(jit, static_argnums=0)
def dimer_probs(lattice, samples):
    '''
    Calculates the probability of presence of monomers, single dimers and double dimers etc on the bulk of the lattice (could be up to four)
    lattice : system on which we want to compute the probabilities
    samples : a bunch of samples over which to compute the probability (...,N)

    return : array of probabilities p=np.array([p_monomer, p_dimer, p_doubledimer, p_triple, p_quadruple)]) (5,)
    '''
    # we only consider the vertices which have 4 atoms -> not on the border
    # for a Torus lattice, the border is empty
    n = len(lattice.vertices) #this container was different for each type of lattice

    # the occupancy of each vertex by sample
    # i.e. number of excited states per vertex
    occupancy = jnp.array([jnp.sum( (1+samples[...,jnp.array(v['atoms'])])/2, axis=-1) for v in lattice.vertices])

    # find out how many of each configuration is present in total
    p0 = jnp.count_nonzero((occupancy-0)==0)/n # no dimer
    p1 = jnp.count_nonzero((occupancy-1)==0)/n # one dimer
    p2 = jnp.count_nonzero((occupancy-2)==0)/n # two dimers
    p3 = jnp.count_nonzero((occupancy-3)==0)/n # three dimers
    p4 = jnp.count_nonzero((occupancy-4)==0)/n # four dimers

    # combine everything and return it normalized to have a probability
    p = jnp.array([p0, p1, p2, p3, p4])

    return p/p.sum()
