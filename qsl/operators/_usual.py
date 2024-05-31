from netket.operator.spin import sigmax,sigmaz, sigmay, sigmam, sigmap
from ._topological import P,Q,R
from ._restricted import _restricted_sigmax, _restricted_sigmay, _restricted_sigmam, _restricted_sigmap

from netket.hilbert import Spin as _SpinHilbert
from ..hilbert import TriangleHilbertSpace as _TriangleHilbertSpace
from netket.operator._local_operator import LocalOperator as _LocalOperator
from netket.utils.types import DType as _DType
from netket.utils.types import Array as _Array
from ..lattice import Kagome as _Kagome

import numpy as np 
import jax.numpy as jnp
from functools import partial
from jax import jit
from typing import Tuple

#######################################################################################################################
################################################## General Operators ##################################################
#######################################################################################################################

def X(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator:
    '''
    Builds the :math:`σ^x=|g><r|+|r><g|` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`
    
    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    if isinstance(hilbert, _TriangleHilbertSpace) or restricted:
        return _restricted_sigmax(hilbert,i)
    else:
        return sigmax(hilbert,i)

def Z(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator:
    '''
    Builds the :math:`σ^z=|g><g|-|r><r|` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    # since it is diagonal, we do not need to define one in the restricted space
    return sigmaz(hilbert,i)

def Y(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator:
    '''
    Builds the :math:`σ^y=i|r><g|-i|g><r|` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    if isinstance(hilbert, _TriangleHilbertSpace) or restricted:
        return _restricted_sigmay(hilbert,i)
    else:
        return sigmay(hilbert,i)

def sigma_minus(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator:
    '''
    Builds the :math:`σ^-=|r><g|` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    if isinstance(hilbert, _TriangleHilbertSpace) or restricted:
        return _restricted_sigmam(hilbert,i)
    else:
        return sigmam(hilbert,i)


def sigma_plus(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator:
    '''
    Builds the :math:`σ^+=|g><r|` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    if isinstance(hilbert, _TriangleHilbertSpace) or restricted:
        return _restricted_sigmap(hilbert,i)
    else:
        return sigmap(hilbert,i)


def g_occ(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator: 
    '''
    Builds the :math:`P_g=|g><g|` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    D = np.zeros((2,2))
    D[0,0] = 1.0
    
    return _LocalOperator(hilbert, D, i)

def r_occ(hilbert:_SpinHilbert, i:int, restricted:bool=True) -> _LocalOperator: 
    '''
    Builds the :math:`P_r=|r><r|=n` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    i : site on which to apply the operator
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)
    
    returns : LocalOperator
    '''
    D = np.zeros((2,2))
    D[1,1] = 1.0
    
    return _LocalOperator(hilbert, D, i)


def r_density(hilbert:_SpinHilbert, lattice:_Kagome, restricted:bool=True) -> _LocalOperator:
    '''
    Builds the :math:`n = 1/N sum_i n_i` operator acting on the i-th site of the (restricted) Hilbert space `hilbert`

    hilbert : hilbert space of the system
    lattice : lattice on which we want the mean (gives the number of sites)
    restricted : flag indicating whether to work in the restricted space (setting 0 outisde the space)

    returns : LocalOperator
    '''
    N = lattice.N

    # The total number of Rydberg excitations on the lattice
    N_op = sum([r_occ(hilbert,i,restricted) for i in range(N)])

    # Mean number of Rydberg exitations
    return N_op/N

def TopoOps(hilbert:_SpinHilbert, lattice:_Kagome, hex:int=0, sites=None) -> Tuple[_LocalOperator,_LocalOperator,_LocalOperator]:
    '''
    Constructs the topological operators on a lattice
    one can either apply on a specific hexagon or on a (list of) site(s)
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
def dimer_probs(lattice:_Kagome, samples:_Array) -> _Array:
    '''
    Calculates the probability of presence of monomers, single dimers and double dimers etc on the bulk of the lattice (could be up to four)
    lattice : system on which we want to compute the probabilities
    samples : a bunch of samples over which to compute the probability (...,N)

    return : array of probabilities p=np.array([p_monomer, p_dimer, p_doubledimer, p_triple, p_quadruple)]) (5,)
    '''
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