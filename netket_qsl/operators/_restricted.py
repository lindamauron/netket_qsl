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
    Builds the :math:`σ^x=|r><g|+|g><r|` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
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

    index = 2**(2-i%3)
    
    # The matrix representing σ^x in the retricted basis (put coefficients outside space to 0)
    D = np.zeros((8,8), dtype=dtype)
    D[0,index] = 1.0
    D[index,0] = 1.0
    
    
    return _LocalOperator(hilbert, D, 3*t+np.arange(3), dtype=dtype)

def _restricted_sigmay(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = complex
) -> _LocalOperator:
    '''
    Builds the :math:`σ^y=i|r><g| - i|g><r|` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''

    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^y)
    t = np.floor(i/3).astype(int) # the triangle index
    
    index = 2**(2-i%3)
    
    # The matrix representing σ^y in the retricted basis (put coefficients outside space to 0)
    D = np.zeros((8,8), dtype=dtype)
    D[0,index] = -1j
    D[index,0] = 1j
    
    
    return _LocalOperator(hilbert, D, 3*t+np.arange(3), dtype=dtype)


def _restricted_sigmam(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = float
) -> _LocalOperator:
    '''
    Builds the :math:`σ^-=|r><g|` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''
    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^+)
    t = np.floor(i/3).astype(int) # the triangle index
    
    index = 2**(2-i%3)
    
    # The matrix representing σ^+ in the retricted basis (put coefficients outside space to 0)
    D = np.zeros((8,8), dtype=dtype)
    D[index,0] = 1.0
    
    return _LocalOperator(hilbert, D, 3*t+np.arange(3), dtype=dtype)



def _restricted_sigmap(
    hilbert: _TriangleHilbertSpace, i: int, dtype: _DType = float
) -> _LocalOperator:
    '''
    Builds the :math:`σ^+=|g><r|` operator acting on the i-th site of the restricted Hilbert space `hilbert`
    
    hilbert: restricted hilbert space
    i: the site on which this operator acts
    dtype: data type of the coefficients
    
    return: a nk.operator.LocalOperator
    '''
    # Find the nieghbors on the same triangle
    # (this is needed to know if we are in the right space after applying σ^-)
    t = np.floor(i/3).astype(int) # the triangle index
    
    index = 2**(2-i%3)
    
    # The matrix representing σ^- in the retricted basis (put coefficients outside space to 0)
    D = np.zeros((8,8), dtype=dtype)
    D[0,index] = 1.0
    
    return _LocalOperator(hilbert, D, 3*t+np.arange(3), dtype=dtype)