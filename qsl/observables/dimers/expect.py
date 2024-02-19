import netket as nk

import jax 
import jax.numpy as jnp

from netket.vqs import MCState, expect
from netket.stats import Stats

from typing import Union

from .operator import DimerProbabilities

from netket.stats.mc_stats import _statistics as mpi_statistics
from functools import partial

@expect.dispatch
def Renyi2(
    vstate: MCState, op: DimerProbabilities, chunk_size: Union[int, None]
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")
    
    samples = vstate.samples
    n_samples = samples.shape[0]*samples.shape[1]
    
    # the occupancy of each vertex by sample
    # i.e. number of excited states per vertex
    occupancy = jnp.array([jnp.sum( (1+samples[...,v])/2, axis=-1) for v in op.v])

    ps = _probs(occupancy,op.n,n_samples)

    # combine everything and return it normalized to have a probability
    p = {'monomer':ps[0],'dimer':ps[1],'double dimer':ps[2],'triple dimer':ps[3],'quadruple dimer':ps[4]}
    return p

@partial(jax.jit, static_argnums=(1,2))
def _probs(occupancy,n_vertices,n_samples):

    # find out how many of each configuration is present in total
    p0 = mpi_statistics(jnp.count_nonzero((occupancy-0)==0)/n_vertices/n_samples+0j) # no dimer
    p1 = mpi_statistics(jnp.count_nonzero((occupancy-1)==0)/n_vertices/n_samples+0j) # one dimer
    p2 = mpi_statistics(jnp.count_nonzero((occupancy-2)==0)/n_vertices/n_samples+0j) # two dimers
    p3 = mpi_statistics(jnp.count_nonzero((occupancy-3)==0)/n_vertices/n_samples+0j) # three dimers
    p4 = mpi_statistics(jnp.count_nonzero((occupancy-4)==0)/n_vertices/n_samples+0j) # four dimers

    return p0,p1,p2,p3,p4