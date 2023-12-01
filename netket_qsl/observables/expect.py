import netket as nk

import jax 
import jax.numpy as jnp

from netket.vqs import MCState, expect
from netket.stats import Stats

from typing import Union

from .operator import Renyi2EntanglementEntropy
from ._renyi2_fcts import _renyi2, _renyi2_bootstrap

@expect.dispatch
def Renyi2(
    vstate: MCState, op: Renyi2EntanglementEntropy, chunk_size: Union[int, None]
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    samples = vstate.samples
    N = samples.shape[-1]
    n_chains = samples.shape[0]
    n_samples_per_chain = samples.shape[1]
    n_samples = n_chains * n_samples_per_chain

    if n_chains % 2 != 0 and not vstate.sampler.is_exact:
        raise ValueError("Use an even number of chains.")

    # for exact samplers, separate in two chains since they are independent anyway
    if n_chains == 1:
        if n_samples % 2 != 0:
            samples = samples[:, :-1]
        
        samples = samples.reshape(2,n_samples_per_chain//2,N)
        n_chains = 2

    # if we don't do bootstrapping, just return the statistics of the samples
    if not op.n_boots:
        σ_η = samples[: (n_chains // 2)]
        σp_ηp = samples[(n_chains // 2) :]


        Renyi2_stats = _renyi2(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            σ_η,
            σp_ηp,
            op.partition,
            chunk_size=chunk_size,
        )
        # Propagation of errors from S_2 to -log(S_2)
        Renyi2_stats = Renyi2_stats.replace(
            variance=Renyi2_stats.variance / (Renyi2_stats.mean.real) ** 2
        )

        Renyi2_stats = Renyi2_stats.replace(
            error_of_mean=jnp.sqrt(
                Renyi2_stats.variance / (n_samples * nk.utils.mpi.n_nodes)
            )
        )

        Renyi2_stats = Renyi2_stats.replace(mean=-jnp.log(Renyi2_stats.mean).real)

        return Renyi2_stats

    # if we do bootstrapping, basically vmap the previous one over all independent (non-correlated) bootstraps
    else :
        entropies =  _renyi2_bootstrap(
            op.rng,
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples,
            op.partition,
            op.n_boots,
            op.chunk_post,
            chunk_size = chunk_size
            )
        op.rng, _ = jax.random.split(op.rng)


        sigma_S = entropies.var()
        return Stats(mean=entropies.mean(), variance=sigma_S, error_of_mean=jnp.sqrt(sigma_S/op.n_boots) ) 

