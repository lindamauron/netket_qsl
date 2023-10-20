import numpy as np 

from ..frequencies import Frequencies
from ._usual import X, r_occ

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
#################################################### Hamiltonians #####################################################
#######################################################################################################################
def potential(hi, lattice, Rb=2.4, Rcut=np.sqrt(7)):
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

    # express the rela length
    Rb *= lattice.a
        
    # range of the interactions
    if Rcut is None:
        Rcut = 100*np.max(lattice.distances) # put it to something bigger than the lattice => all interactions are taken
    else:
        Rcut *= lattice.a
        
        
    V = 0
    d = lattice.distances

    # sum over all pairs
    for i in range(N):
        for j in range(i+1,N):
            
            # cutting radius
            if d[i,j] <= Rcut:
                V += r_occ(hi,i)*r_occ(hi,j) * (Rb/d[i,j])**6
                    
    return V

class Hamiltionian:
    '''
    Constructs the support fot the hamiltonian's operator. The Hamiltonian is of the form : 
    H = -Ω(t)/2 \sum_i X_i - Δ(t) \sum_i n_i + Ω(t)/2 \sum_ij (Rb/r_ij)^6 n_i n_j
    where :
        - the schedule of Ω(t), Δ(t) is entirely defined by the sweep_time
        - Rb is a parameter usually set to 3th neighbor
        - the potential can be cut up to a certain distance (Rcut)

    Hamiltonian(t) can be called
    '''
    def __init__(self, hi, lattice, sweep_time=2.5, Rb=2.4, Rcut=None):
        '''
        Constructs the hamiltonian instance, with a rydberg blockade at Rb, interactions up to Rcut
        hi : hilbert space of the system
        lattice : lattice on which the operator should act
        sweep_time : whole duration of the time evolution
        Rcut : range of the potential
                if r_ij > Rcut, V_ij = 0
        Rb : Rydberg blockade radius in units of a (lattice vector)
        '''
        # Infos of the operator
        self.str = f'Hamiltonian({lattice}, T={sweep_time}, Rb={Rb}, Rcut={Rcut})'

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
        self.R = R

        N = lattice.N

        # The total number of Rydberg excitations on the lattice
        self.N_op = sum([r_occ(hi,i) for i in range(N)])

        # Sum of X operators on the whole lattice (so it is not computed in each loop)
        self.Xtot_op = sum([X(hi,i) for i in range(N)])

        # Potential
        self.V_op = potential(hi, lattice, Rb, Rcut)

        # Frequency schedules to be determined only once
        self.frequencies = Frequencies(sweep_time)

    def __repr__(self):
        '''
        Representation of the class
        '''
        return self.str


    def __call__(self, t):
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
    
    def delta_h(self, d):
        '''
        H(Δ(t)) : computes the LocalOperator Hamiltonian at time t such that Δ(t) = d
        d : relative value of Δ(t), i.e. d = Δ/Ωf

        returns : LocalOperator H
        '''

        return self.frequencies.Ωf*( -1/2*self.Xtot_op - d*self.N_op + self.V_op )

    
    def operators(self):
        '''
        Returns each individual operator, independent of time

        returns : (\sum_i X_i , \sum_i n_i , 1/2 \sum_ij (Rb/r_ij)^6 n_i n_j ) LocalOperators
        '''
        return self.Xtot_op, self.N_op, self.V_op
    

    def for_sparse(self):
        '''
        Returns each individual operator, independent of time, as a sparse matrix

        returns : (\sum_i X_i , \sum_i n_i , 1/2 \sum_ij (Rb/r_ij)^6 n_i n_j ) SparseMatrices
        '''
        return self.Xtot_op.to_sparse(), self.N_op.to_sparse(), self.V_op.to_sparse()
