import numpy as np 

from ..hilbert import TriangleHilbertSpace as _TriangleHilbertSpace
from netket.operator._local_operator import LocalOperator as _LocalOperator
from netket.utils.types import DType as _DType



#######################################################################################################################
################################################## General Operators ##################################################
#######################################################################################################################
def _restricted_sigmax(
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

def _restricted_sigmay(
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
    # (this is needed to know if we are in the right space after applying σ^y)
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



def _restricted_sigmap(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = float
) -> _LocalOperator:
    '''
    Builds the :math:`σ^+=|r><g|` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''
    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^p)
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
    
    return _LocalOperator(hilbert, D, (i,j,k), dtype=dtype)

def _restricted_sigmam(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = float
) -> _LocalOperator:
    '''
    Builds the :math:`σ^-=|g><r|` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''
    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^p)
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
    D[index,0] = 1.0    
    
    return _LocalOperator(hilbert, D, (i,j,k), dtype=dtype)