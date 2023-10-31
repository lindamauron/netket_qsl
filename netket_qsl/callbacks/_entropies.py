import numpy as np

import netket as nk

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit

from scipy.special import logsumexp


#######################################################################################################################
################################################## Callback functions #################################################
#######################################################################################################################


class callback_entropy:
    def __init__(self, lattice, sites, name=None):
        N = lattice.N
        mask_A = np.zeros((N,), dtype=bool)
        mask_A[np.array(sites)] = True

        mask_B = np.logical_not(mask_A)
        
        self.mask_A = mask_A
        self.mask_B = mask_B
        
        if name is None:
            name = 'S'
        self.name = name
        
    def __call__(self, step, log_data, driver):
        s1 = driver.state.sample()
        N = s1.shape[-1]

        s1 = s1.reshape(-1, N)
        s2 = driver.state.sample().reshape(-1, N)

        Ns = s1.shape[0]
        
        
        s12 = (s1*self.mask_A + s2*self.mask_B)
        s21 = (s2*self.mask_A + s1*self.mask_B)
        #s3 = (s1*self.mask_A + s1*self.mask_B)
        #s4 = (s2*self.mask_A + s2*self.mask_B)

        s = (driver.state.log_value(s12)-driver.state.log_value(s1)) + (driver.state.log_value(s21)-driver.state.log_value(s2))

        log_data[self.name] = -logsumexp( s ) + np.log(Ns)
        
        return True
    
def random_even_bipartition(key, size):
    # divides the samples into two random chains
    permutation = jax.random.permutation(key, jnp.arange(size))
    half = size // 2
    return permutation[:half], permutation[half:]



class bootstrap_entropy:
    def __init__(self, lattice, sites, n_boots, name=None):
        N = lattice.N
        
        self.mask_A = jnp.zeros((N,), dtype=bool).at[jnp.array(sites)].set(True)
        self.mask_B = jnp.logical_not(self.mask_A)

        if name is None:
            name = 'S'
        self.name = name

        self.n_boots = n_boots

    
    def __call__(self, key, vs, log_data):
        #sample once and reshape everything
        samples = vs.sample()
        N = samples.shape[-1]
        samples = samples.reshape(-1, N)
        Ns = samples.shape[0]

        # we effectively split the samples into two random partitions, different for each bootstrap
        kps = rnd.split(key, self.n_boots)
        part1, part2 = jax.vmap(random_even_bipartition, (0, None))(kps, Ns)
        N_2 = Ns // 2
        # the log values can be stored since we will pick from these states
        lv = vs.log_value(samples)
        
        # select the bootstrap indices
        key, _ = rnd.split(key,2)
        b_idcs = rnd.randint(key, shape=(self.n_boots, N_2), minval=0, maxval=N_2)
        
        # combine them with the partitions to know what sample indices we will use
        f = jax.vmap(lambda partition, idcs: partition[idcs])
        bidcs1 = f(part1, b_idcs)
        bidcs2 = f(part2, b_idcs)

        @jit
        def f(samples, k):
            # computes the entropy of the k-th bootstrap
            x12 = samples[bidcs1[k]]*self.mask_A + samples[bidcs2[k]]*self.mask_B
            x21 = samples[bidcs2[k]]*self.mask_A + samples[bidcs1[k]]*self.mask_B

            lv12 = vs.log_value(x12)
            lv21 = vs.log_value(x21)

            return samples, -jax.scipy.special.logsumexp(lv12 + lv21 - lv[bidcs1[k]] - lv[bidcs2[k]]) + jnp.log(N_2)
        
        # loop over all bootsraps and get the statisitcs
        samples, ents = jax.lax.scan(f, samples, jnp.arange(self.n_boots) )
        log_data[self.name] = nk.stats.statistics(ents)

        return True
    
