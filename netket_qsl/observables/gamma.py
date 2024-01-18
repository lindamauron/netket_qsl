import numpy as np
import jax.numpy as jnp
from typing import Union
from netket.hilbert import Spin as _SpinHilbert

from .entropy import Renyi2EntanglementEntropy as Renyi2
class Gamma:
    def __init__(self, 
                 hilbert: _SpinHilbert,
                 partitionA: jnp.array,
                 partitionB: jnp.array,
                 partitionC: jnp.array,
                 n_boots: Union[int, None] = None,
                 seed: Union[int, None] = None,
                 chunk_post: Union[int, None] = None
                 ):
        self._hilbert = hilbert
        SA = Renyi2(hilbert,partitionA,n_boots,seed,chunk_post)
        SB = Renyi2(hilbert,partitionB,n_boots,seed,chunk_post)
        SC = Renyi2(hilbert,partitionC,n_boots,seed,chunk_post)


        SAB = Renyi2(hilbert,np.append(partitionA,partitionB),n_boots,seed,chunk_post)
        SBC = Renyi2(hilbert,np.append(partitionB,partitionC),n_boots,seed,chunk_post)
        SCA = Renyi2(hilbert,np.append(partitionC,partitionA),n_boots,seed,chunk_post)

        SABC = Renyi2(hilbert,np.append(partitionA,np.append(partitionB,partitionC)),n_boots,seed,chunk_post)

        self.S = {'A':SA, 'B':SB, 'C':SC, 'AB':SAB, 'BC':SBC, 'CA':SCA, 'ABC':SABC}

    @property
    def hilbert(self):
        r"""The hilbert space associated to this observable."""
        return self._hilbert


    def partition(self,x) :
        r"""
        list of indices for the degrees of freedom in the partition x
        """
        return self.S[x].partition
    
    def draw(self):
        state = np.zeros(self.hilbert.size, dtype=int)

        for i,x in enumerate(['A','B','C']):
            part = self.partition(x)
            state[part] = i+1
        
        colors = ['k', 'r', 'g', 'b']
        return state, colors

    def __repr__(self):
        return f"gamma(hilbert={self.hilbert}, tri-partition=({list(self.partition('A')),list(self.partition('B')),list(self.partition('C'))})"
    
    def _reset(self, seed: Union[int, None] = None):
        for s in self.S.values:
            s.reset(seed)

        return    

import netket as nk
from netket.stats import Stats
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt
from .entropy._renyi2_fcts import _renyi2, _renyi2_bootstrap
import jax.random as rnd

class callback_gamma_boots:
    def __init__(self,gamma,folder,plot=False):
        self.gamma = gamma
        self.folder = folder
        self.plot = plot
            

    def __call__(self,step,log_datas,vs):
        log_boots, log_noboot = log_datas

        samples = vs.samples
        N = samples.shape[-1]
        n_chains = samples.shape[0]
        n_samples_per_chain = samples.shape[1]
        σ_η = samples[: (n_chains // 2)].reshape(-1,N)
        σp_ηp = samples[(n_chains // 2) :].reshape(-1,N)

        entropies = {}
        for S in self.gamma.S:
            entropies[S] = {}

        for S in self.gamma.S:
            # no bootstrapping
            Renyi2_stats = _renyi2(
                vs._apply_fun,
                vs.parameters,
                vs.model_state,
                σ_η,
                σp_ηp,
                self.gamma.S[S].partition,
                chunk=vs.chunk_size,
            )
            # Propagation of errors from S_2 to -log(S_2)
            Renyi2_stats = Renyi2_stats.replace(
                variance=Renyi2_stats.variance / (Renyi2_stats.mean.real) ** 2
            )

            n_samples = n_chains * n_samples_per_chain
            Renyi2_stats = Renyi2_stats.replace(
                error_of_mean=np.sqrt(
                    Renyi2_stats.variance / (n_samples * nk.utils.mpi.n_nodes)
                )
            )

            Renyi2_stats = Renyi2_stats.replace(mean=-np.log(Renyi2_stats.mean) )

            log_noboot['S'+S] = Renyi2_stats


            # bootstrapping
            entropies[S] =  nk.jax.apply_chunked(_renyi2_bootstrap, in_axes=(None,None,None,None,0,0,None,None,None,None), chunk_size=self.gamma.S[S].chunk_post)(
                self.gamma.S[S].rng,
                vs._apply_fun,
                vs.parameters,
                vs.model_state,
                σ_η,
                σp_ηp,
                self.gamma.S[S].partition,
                self.gamma.S[S].n_boots,
                self.gamma.S[S].chunk_post,
                chunk = vs.chunk_size
                )
                
            self.gamma.S[S].rng, _ = rnd.split(self.gamma.S[S].rng)
            mean_S = entropies[S].mean()
            sigma_S = entropies[S].real.var()

            log_boots['S'+S] = Stats(mean=mean_S,
                            variance=sigma_S,
                            error_of_mean=np.sqrt(sigma_S/self.gamma.S[S].n_boots),
                        )
                            
        # Gamma
        gammas = -(entropies['A'] + entropies['B'] + entropies['C']
                    - entropies['AB'] - entropies['BC'] - entropies['CA']
                    + entropies['ABC']
                    )
        mean_g = gammas.mean()
        sigma_g = gammas.real.var()

        log_boots['γ'] = Stats(mean=mean_g, variance=sigma_g, error_of_mean=np.sqrt(sigma_g/self.gamma.S['A'].n_boots))
        callback_gamma(step,log_noboot,vs)

        if self.plot and rank==0:
            for S in self.gamma.S:
                plt.figure(figsize=(6,3))
                plt.subplot(1,2,1)
                plt.hist(entropies[S].real)
                plt.axvline(log_boots['S'+S].mean.real, ls='-', color='k')
                plt.axvline(log_boots['S'+S].mean.real+np.sqrt(log_boots['S'+S].variance), ls=':', color='k')
                plt.axvline(log_boots['S'+S].mean.real-np.sqrt(log_boots['S'+S].variance), ls=':', color='k')
                plt.axvline(log_noboot['S'+S].mean.real, ls='-', c='r')
                plt.xlabel(r'Re [ $S_2$ ]')
                plt.ylabel(r'Counts')

                plt.subplot(1,2,2)
                plt.hist(entropies[S].imag)
                plt.xlabel(r'Im [ $S_2$ ]')
                plt.ylabel(r'Counts')

                plt.tight_layout()
                plt.savefig(self.folder+'_S'+S+f'_t={step:.2f}.png')
                plt.close()


            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.hist(gammas.real)
            plt.axvline(mean_g.real, ls='-', color='k')
            plt.axvline(mean_g.real+np.sqrt(sigma_g), ls=':', color='k')
            plt.axvline(mean_g.real-np.sqrt(sigma_g), ls=':', color='k')
            plt.axvline(log_noboot['γ'].mean.real, ls='-', c='r')
            plt.xlabel(r'Re [ $\gamma$ ]')
            plt.ylabel(r'Counts')

            plt.subplot(1,2,2)
            plt.hist(gammas.imag)
            plt.xlabel(r'Im [ $\gamma$ ]')
            plt.ylabel(r'Counts')

            plt.tight_layout()
            plt.savefig(self.folder+'_'+'gamma'+f'_t={step:.2f}.png')
            plt.close()
        
        return True
        

def callback_gamma(step,log_data,vs):
    gamma = -(log_data['SA'].mean + log_data['SB'].mean + log_data['SC'].mean
                - log_data['SAB'].mean - log_data['SBC'].mean - log_data['SCA'].mean
                + log_data['SABC'].mean
                )
    err = (log_data['SA'].error_of_mean + log_data['SB'].error_of_mean + log_data['SC'].error_of_mean 
           + log_data['SAB'].error_of_mean + log_data['SBC'].error_of_mean + log_data['SCA'].error_of_mean
           + log_data['SABC'].error_of_mean 
           )
    
    var = np.sqrt([log_data['SA'].variance, log_data['SB'].variance, log_data['SC'].variance,
           log_data['SAB'].variance, log_data['SBC'].variance, log_data['SCA'].variance,
           log_data['SABC'].variance]
           ).sum()
    
    log_data['γ'] = Stats(mean=gamma, error_of_mean=err, variance=var)
    return True