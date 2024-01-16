import numpy as np
import orjson

def append_logs(log1, log2, 
                operators=['Generator', 'n', 'P', 'Q', 'R'], 
                callbacks=['acc', 'Delta', 'Omega', 't', 'monomer', 'dimer', 'double dimer', 'triple dimer', 'quadruple dimer'] 
               ):
    '''
    Creates a new log where log2 has been appended to log1. This results in a log witha ll the same entries, but lists are appended to follow. 
    All entries not entered as arguments are not listed in the resulting log. 
    log1, log2 : logs
    operators : operator entries of the logs
    callbacks : other kind of entries in the logs
    
    return : log with entries operators and callbacks
    '''
    new_log = {}
    for k in operators:
        new_log[k] = {}
        new_log[k]['iters'] = list( np.append(log1[k]['iters'], log2[k]['iters']) )
        
        new_log[k]['Mean'] = {}
        new_log[k]['Mean']['real'] = list( np.append( log1[k]['Mean']['real'], log2[k]['Mean']['real']) )
        new_log[k]['Mean']['imag'] = list( np.append( log1[k]['Mean']['imag'], log2[k]['Mean']['imag']) )

        new_log[k]['Variance'] = list( np.append( log1[k]['Variance'], log2[k]['Variance']) )
        new_log[k]['Sigma'] = list( np.append( log1[k]['Sigma'], log2[k]['Sigma']) )
        new_log[k]['R_hat'] = list( np.append( log1[k]['R_hat'], log2[k]['R_hat']) )
        new_log[k]['TauCorr'] = list( np.append( log1[k]['TauCorr'], log2[k]['TauCorr']) )

    for k in callbacks:
        new_log[k] = {}
        new_log[k]['iters'] = list( np.append( log1[k]['iters'], log2[k]['iters']) )
        new_log[k]['value'] = list( np.append( log1[k]['value'], log2[k]['value']) )
    
    return new_log

def save_log(filename,log):
    '''
    saves the log as a filename.log file
    '''
    def _serialize(log, outstream):
        r"""
        Inner method of `serialize`, working on an IO object.
        """
        outstream.write(
            orjson.dumps(
                log,
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )

    with open(filename+'.json', "wb") as io:
        _serialize(log, io)

    return
    

#######################################################################################################################
#################################################### Exact vectors ####################################################
#######################################################################################################################
def tohilb(psi):
    '''
    Transforms a pm 1 state to a full hilbert space vector
    psi : ndarray (N,)

    returns : ndarray (2**N,)
    '''
    N = psi.shape[-1]
    coeffs = (1+psi)/2

    v = coeffs[-1::-1]*2**np.arange(psi.shape[-1])

    new = np.zeros(2**N)
    new[int(v.sum())] = 1

    #print('out is ', new)
    return new


def tonk(psi):
    '''
    Transforms a full hilbert space vector to a pm 1
    psi : ndarray (2**N,)

    returns : ndarray (N,)
    '''
    N = np.log2(psi.shape[-1], dtype=int)
    i = np.where(psi)[0][0]
    #print(i)

    coeff = np.zeros(N, bool)
    for k in range(N):
        coeff[N-k-1] = i%2
        i = i//2
	    
	#print(coeff)
	    

    new = -np.ones(N)
    new[coeff] = 1
    return new



def sort_W(W, lattice):
    '''
    Sorts the matrix of weights W in terms of distance between the sites. 
    This means that the arguments W'_i0 = W_ii, W'_ik is the k-the neighbor of i (keep in mind that the neighbors have degeneracy). 
    Be careful, on certain lattices, the neighbors are not identical per sites, due to pbc.
    W : matrix of weights (N,N)
    lattice : lattice on which our system lives

    returns : sorted W' (N,N)
    '''
    dis = np.rint((lattice.distances/lattice.a)**2)
    args = np.argsort(dis, axis=-1)

    return np.take_along_axis(W, args, axis=-1)


################### MPI functionalities ###################
from mpi4py import MPI
comm = MPI.COMM_WORLD
def mpi_print(*values, sep=' ', end='\n', file=None, flush=False, only_once=True) -> None:
    '''
    Allows to print information on one from one single mpi rank, avoiding the repetition.
    There is also the posbility to print on each rank, specifying the rank for each print.
    only_once: bool indicating whether one wants to print only on rank 0 or on all ranks
    This works as the usual print function.

    returns: print(*values)
    '''
    rank = comm.Get_rank()

    if only_once and rank==0:
        return print(*values, sep=sep, end=end, file=file, flush=flush)
    elif not only_once:
        values = (f'Rank {rank} : ',) + values
        return print(*values, sep=sep, end=end, file=file, flush=flush)

    return