import jax.numpy as jnp

from qsl.observables.entropy._renyi2_fcts import _renyi2_bootstrap as bootstrap_entropies
from netket.stats import Stats


#######################################################################################################################
################################################## Callback functions #################################################
#######################################################################################################################

from mpi4py import MPI
import matplotlib.pyplot as plt


class callback_entropy_distribution:
    def __init__(self,operator,folder='',name='S',save_log=False):
        self.op = operator
        self.folder = folder
        self.name = name
        self.save = save_log

    def __call__(self,step,log_data,vstate):
        
        entropies = bootstrap_entropies(
            self.op.rng,
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            vstate.samples,
            self.op.partition,
            self.op.n_boots,
            self.op.chunk_post,
            chunk_size = vstate.chunk_size
            )
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank==0:
            mean_S = entropies.mean()
            sigma_S = jnp.sqrt(entropies.var())

            if self.save:
                log_data[self.name] = Stats(mean=mean_S, variance=sigma_S**2, error_of_mean=sigma_S/jnp.sqrt(self.op.n_boots) ) 

        
            plt.figure(figsize=(6,3))

            plt.subplot(1,2,1)
            plt.hist(entropies.real)
            plt.axvline(mean_S.real, ls='-', color='k')
            plt.axvline(mean_S.real+sigma_S, ls=':', color='k')
            plt.axvline(mean_S.real-sigma_S, ls=':', color='k')
            plt.xlabel('real part of S_2')
            plt.ylabel('counts')

            plt.subplot(1,2,2)
            plt.hist(entropies.imag)
            plt.xlabel('imag part of S_2')
            plt.ylabel('counts')

            plt.tight_layout()
            plt.savefig(self.folder+f'{step}.png')
            plt.close()
        
        return True