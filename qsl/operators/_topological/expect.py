import netket as nk

import jax.numpy as jnp
from typing import  Optional

from netket.stats import Stats
from netket.stats.mc_stats import _statistics as mpi_statistics

from .base import TopoOperator, get_local_kernel_arguments, o_loc, sparsify



@nk.vqs.expect.dispatch(precedence=10)
def expect_topoOp(vstate: nk.vqs.MCState, op:TopoOperator, chunk_size: Optional[int]):
    sigma, extra_args = get_local_kernel_arguments(vstate, op, chunk_size)

    n_chains = sigma.shape[0]
    N = sigma.shape[-1]
    sigma = sigma.reshape(-1, N)
    etas = extra_args[0].reshape(-1,N)
    mels = extra_args[1].reshape(-1)

    # eta, mels = extra_args
    E_loc = o_loc(vstate._apply_fun, vstate.variables, sigma, (etas,mels), chunk_size).reshape(n_chains,-1)

    return mpi_statistics(E_loc)


@nk.vqs.expect.dispatch(precedence = 10)
def expect_topoOp(vstate: nk.vqs.FullSumState, Ô: TopoOperator) -> Stats:

    O = sparsify(Ô)
    Ψ = vstate.to_array()

    # TODO: This performs the full computation on all MPI ranks.
    # It would be great if we could split the computation among ranks.

    OΨ = O @ Ψ
    expval_O = (Ψ.conj() * OΨ).sum()

    variance = jnp.sum(jnp.abs(OΨ - expval_O * Ψ) ** 2)
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)

