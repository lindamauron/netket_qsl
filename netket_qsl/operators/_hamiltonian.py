import numpy as np 

from typing import Tuple
from netket.utils.types import Array
from scipy.sparse import csr_matrix as _csr_matrix
from netket.operator import DiscreteOperator as _DiscreteOperator

from ..frequencies import Frequency as _Frequency
from ..frequencies import Cubic
from ._usual import X, r_occ
from ..lattice import Kagome as _Kagome
from netket.hilbert import Spin as _SpinHilbert


#######################################################################################################################
#################################################### Hamiltonians #####################################################
#######################################################################################################################
def vdW_potential(hi:_SpinHilbert, lattice:_Kagome, Rb:float=2.4, Rcut:float=np.sqrt(7)) -> Tuple[_DiscreteOperator,Array]:
    '''
    Returns the potential operator of the lattice i.e. V/Ω = sum (Rb/rij)^6 ni nj so that there is no dependence on time
    with a rydberg blockade at Rb, interactions up to Rcut
    hi : hilbert space of the system
    lattice : lattice on which the operator should act
    Rb : Rydberg blockade radius in units of a (lattice vector)
    Rcut : range of the potential
            if r_ij > Rcut, V_ij = 0

    returns : LocalOperator corresponding to the full potential term V/Ω 
    '''        
    N = lattice.N 

    # express the real length
    Rb *= lattice.a
        
    # range of the interactions
    if Rcut is None:
        Rcut = 100*np.max(lattice.distances) # put it to something bigger than the lattice => all interactions are taken
    else:
        Rcut *= lattice.a

    # Construct the interaction matrix by precomputing the distances ratio up to a cut-off
    # we define our matrix R_ij = (Rb/rij)^6 for rij < Rcut but with zeros on the diagonal 
    # it will be used for every update, so we do not compute it every time
    R = lattice.distances
    np.fill_diagonal(R, 1)
    R = (Rb/R)**6
    R[lattice.distances>Rcut] = 0
    np.fill_diagonal(R, 0)
    R = R    

    V = 0
    d = lattice.distances

    # sum over all pairs
    for i in range(N):
        for j in range(i+1,N):
            
            # cutting radius
            if d[i,j] <= Rcut:
                V += r_occ(hi,i)*r_occ(hi,j) * (Rb/d[i,j])**6
                    
    return V, R

class Rydberg_Hamiltionian:
    '''
    Constructs the support fot the hamiltonian's operator. The Hamiltonian is of the form : 
    H = -Ω(t)/2 \sum_i X_i - Δ(t) \sum_i n_i + Ω(t)/2 \sum_ij (Rb/r_ij)^6 n_i n_j
    where :
        - the schedule of Ω(t), Δ(t) is entirely defined by the sweep_time
        - Rb is a parameter usually set to 3th neighbor
        - the potential can be cut up to a certain distance (Rcut)

    Hamiltonian(t) can be called
    '''
    def __init__(self, hi:_SpinHilbert, lattice:_Kagome, frequencies:_Frequency=Cubic(2.5), Rb:float=2.4, Rcut:float=None):
        '''
        Constructs the hamiltonian instance, with a rydberg blockade at Rb, interactions up to Rcut
        hi : hilbert space of the system
        lattice : lattice on which the operator should act
        frequencies : callable giving both frequencies Ω, Δ at a given time
        Rcut : range of the potential
                if r_ij > Rcut, V_ij = 0
        Rb : Rydberg blockade radius in units of a (lattice vector)
        '''
        N = lattice.N

        # The total number of Rydberg excitations on the lattice
        self.N_op = sum([r_occ(hi,i) for i in range(N)])

        # Sum of X operators on the whole lattice (so it is not computed in each loop)
        self.Xtot_op = sum([X(hi,i) for i in range(N)])

        # Potential
        self.V_op, self.R = vdW_potential(hi, lattice, Rb, Rcut)

        # Frequency schedules to be determined only once
        self.frequencies = frequencies

        # Infos of the operator
        self._str = f'Hamiltonian({lattice}, (Ω,Δ)={frequencies}, Rb={Rb}, Rcut={Rcut})'

    def __repr__(self):
        '''
        Representation of the class
        '''
        return self._str


    def __call__(self, t:float) -> _DiscreteOperator:
        '''
        H(t) : computes the LocalOperator Hamiltonian at time t
        t : time

        returns : LocalOperator H(t)
        '''
        # the frequencies at the interesting time
        t = np.array(t)
        Ω, Δ = self.frequencies(t)

        # our operator
        return -Ω/2*self.Xtot_op - Δ*self.N_op + Ω*self.V_op
    
    def of_delta(self, d:float) -> _DiscreteOperator:
        '''
        H(Δ(t)) : computes the LocalOperator Hamiltonian at time t such that Δ(t) = d
        d : relative value of Δ(t), i.e. d = Δ/Ωf

        returns : LocalOperator H
        '''

        return self.frequencies.Ωf*( -1/2*self.Xtot_op - d*self.N_op + self.V_op )

    
    def operators(self) -> Tuple[_DiscreteOperator,_DiscreteOperator,_DiscreteOperator]:
        '''
        Returns each individual operator, independent of time

        returns : (\sum_i X_i , \sum_i n_i , 1/2 \sum_ij (Rb/r_ij)^6 n_i n_j ) LocalOperators
        '''
        return self.Xtot_op, self.N_op, self.V_op
    

    def for_sparse(self) -> Tuple[_csr_matrix,_csr_matrix,_csr_matrix]:
        '''
        Returns each individual operator, independent of time, as a sparse matrix

        returns : (\sum_i X_i , \sum_i n_i , 1/2 \sum_ij (Rb/r_ij)^6 n_i n_j ) SparseMatrices
        '''
        return self.Xtot_op.to_sparse(), self.N_op.to_sparse(), self.V_op.to_sparse()

    def as_lists(self) -> Tuple[Tuple[_csr_matrix,_csr_matrix,_csr_matrix], Tuple[callable,callable,callable]]:
        '''
        Generates two lists of mutually callables which work as follows : 
        H(t) = sum([f(t)*op for op,f in zip(ops,freqs) ])

        returns : (sparse operators, callable coefficients)
        '''
        ops = self.for_sparse()
        freqs = (lambda t : -self.frequencies.Ω(t)/2, lambda t : -self.frequencies.Δ(t), lambda t : self.frequencies.Ωf)

        return ops, freqs