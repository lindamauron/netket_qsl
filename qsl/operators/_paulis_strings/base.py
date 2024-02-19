import numpy as np
import netket as nk

from .logic import simplify, addition, multiplication

class PauliStringConstructor:
    '''
    Generates operators as Pauli strings. Each instance has a list of Pauli operators [I,X,Y,Z] and a list of corresponding coefficients. 
    The goal of this class is to provide with an effective constructor for pauli strings, allowing for basic operations and simplifications of pauli strings. 
    
    Notice that it has not been fully tested and only works ok for precise cases : to be perfected.
    '''
    def __init__(self, hilbert, Op=None):
        '''
        hi : hilbert space
        Op : tuple of operator strings and weights
        '''
        self.hi = hilbert
        
        if Op is not None:
            strings, weights = simplify(Op)
        else:
            strings = [hilbert.size*'I']
            weights = [0]
        self.strings = np.array(strings, dtype=f'<U{hilbert.size}')
        self.weights = np.array(weights, dtype=complex)
        
        if self.strings.shape[0]!=self.weights.shape[0]:
            raise ValueError( 
                f"The number of operators must be the same as the number of coeffs, but got {self.weights.shape[0]} and {self.strings.shape[0]}"
            )

    def __call__(self):
        return self.strings, self.weights

    def to_nk(self):
        '''
        Generates the netket operator corresponding to this instance
        '''
        return nk.operator.PauliString(self.hi,self.strings,self.weights)
        
        
    def __mul__(self, other):
        '''
        Multiplication between instances or with scalars.
        '''
        if isinstance(other, PauliStringConstructor):
            assert self.hi == other.hi, 'The hilbert spaces must be identical'
            return PauliStringConstructor(self.hi, multiplication( self(), other() ) )
        
        if np.issubdtype(type(other), np.number):
            if other == 1.0:
                return self
            return PauliStringConstructor(self.hi, (self.strings, other*self.weights) )
        else:
            return NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, PauliStringConstructor):
            return other.__mul__(self)
        
        if np.issubdtype(type(other), np.number):
            if other == 1.0:
                return self
            return self.__mul__(other)
        else:
            return NotImplemented
        
    def __add__(self, other):
        if isinstance(other, PauliStringConstructor):
            assert self.hi == other.hi, 'The hilbert spaces must be identical'
            return PauliStringConstructor(self.hi, addition( self(), other() ) )
        
        if np.issubdtype(type(other), np.number):
            if other == 0.0:
                return self
            else:
                otherOp = other*I(self.hi)
                return self.__add__(otherOp)
        else:
            return NotImplemented
        
    def __radd__(self, other):
        return self+other
        
    def __sub__(self, other):
        if isinstance(other, PauliStringConstructor):
            return self.__add__(-1.0*other)
        
        if np.issubdtype(type(other), np.number):
            if other == 0:
                return self
            else:
                otherOp = other*I(self.hi)
                return self.__add__(otherOp)
        else:
            return NotImplemented
        
    def __neg__(self):
        return -1.0*self
        
    def __repr__(self):
        print_list = []
        for op, w in zip(self.strings, self.weights):
            print_list.append("    {} : {}".format(op, str(w)))
        s = "PauliStringConstructors(hilbert={}, n_strings={}, dict(operators:weights)=\n{}\n)".format(
            self.hi, len(self.strings), ",\n".join(print_list)
        )
        return s
        
        
class I(PauliStringConstructor):
    '''
    Identity operator
    '''
    def __init__(self, hi):
        
        N = hi.size
    
        string = N*'I'
        weights = [1]
        super().__init__(hi, (string,weights) )