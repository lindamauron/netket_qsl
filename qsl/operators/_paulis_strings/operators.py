import numpy as np
from .base import PauliStringConstructor   
        
class gg(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for |g><g| acting on site i of hilbert space hi
    '''
    def __init__(self, hi, i):
        
        N = hi.size

        string = np.char.add( np.char.add( i*'I', ['I', 'Z'] ), (N-i-1)*'I')
        weights = [1/2, 1/2]
        super().__init__(hi, (string,weights) )
        
class ee(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for |e><e| acting on site i of hilbert space hi
    '''
    def __init__(self, hi, i):
        
        N = hi.size

        string = np.char.add( np.char.add( i*'I', ['I', 'Z'] ), (N-i-1)*'I')
        weights = [1/2, -1/2]
        super().__init__(hi, (string,weights) )
        
class sp(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for |e><g| acting on site i of hilbert space hi
    '''
    def __init__(self, hi, i):
        
        N = hi.size

        string = np.char.add( np.char.add( i*'I', ['X', 'Y'] ), (N-i-1)*'I')
        weights = [1/2, 1j/2]
        super().__init__(hi, (string,weights) )
        
class sm(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for |g><e| acting on site i of hilbert space hi
    '''
    def __init__(self, hi, i):
        
        N = hi.size

        string = np.char.add( np.char.add( i*'I', ['X', 'Y'] ), (N-i-1)*'I')
        weights = [1/2, -1j/2]
        super().__init__(hi, (string,weights) )
        
class X(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for |g><e|+|e><g| acting on site i of hilbert space hi
    '''
    def __init__(self, hi, i):
        
        N = hi.size
    
        string = np.char.add( np.char.add( i*'I', ['X'] ), (N-i-1)*'I')
        weights = [1]
        super().__init__(hi, (string,weights) )
        
class Y(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for i|g><r|-i|r><g| acting on site i of hilbert space hi
    '''

    def __init__(self, hi, i):
        
        N = hi.size
    
        string = np.char.add( np.char.add( i*'I', ['Y'] ), (N-i-1)*'I')
        weights = [1]
        super().__init__(hi, (string,weights) )
        
class Z(PauliStringConstructor):
    '''
    Constuctor of the Pauli string for |g><g|-|e><e| acting on site i of hilbert space hi
    '''
    def __init__(self, hi, i):
        
        N = hi.size
    
        string = np.char.add( np.char.add( i*'I', ['Z'] ), (N-i-1)*'I')
        weights = [1]
        super().__init__(hi, (string,weights) )
