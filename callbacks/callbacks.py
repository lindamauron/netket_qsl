import numpy as np
import matplotlib.pyplot as plt

import tensorboardX as tbx

import netket as nk

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit

from scipy.special import logsumexp

from ..operators import dimer_probs
from ..lattice import Kagome as Lattice

#######################################################################################################################
################################################## Callback functions #################################################
#######################################################################################################################


def cb_params(step, log_data, driver):
    pars = driver.state.parameters
    log_data['pars'] = np.fromiter(pars.values(), dtype=float)
    #print(step, '\t', driver.state.parameters)

    return True


def callback_acc(step, log_data, driver):
    '''
    Acceptance of the sampler during the evolution
    '''
    log_data['acc'] = driver.state.sampler_state.acceptance 
    
    return True

def callback_omega_delta(step, log_data, driver):
    '''
    stores the values of Ω(t) and Δ(t) used during the evolution
    Note : the values stored are the frequency/Ω
    '''
    O, D = driver.generator.frequencies(driver.t)

    log_data['Omega'] = O/driver.generator.frequencies.Ωf
    log_data['Delta'] = D/driver.generator.frequencies.Ωf
    return True

class callback_dimerprobs:
    def __init__(self,lattice:Lattice):
        self.lattice = lattice

    def __call__(self, step, log_data, driver):
        probs = dimer_probs(self.lattice, driver.state.samples)
        log_data['monomer'] = probs[0]
        log_data['dimer'] = probs[1]
        log_data['double dimer'] = probs[2]
        log_data['triple dimer'] = probs[3]
        log_data['quadruple dimer'] = probs[4]

        return True

class callback_dimerprobs_MF:
    def __init__(self,lattice:Lattice):
        self.lattice = lattice

    def __call__(self, step, log_data, driver):
        phi = driver.state.parameters['ϕ']

        a = phi[:,0]
        b = phi[:,1]

        ni = (b.conj()*b).real

        #n = len(lattice.non_border) #this container was different for each type of lattice
        vertices = self.lattice.vertices[self.lattice.non_border]

        # find out how many of each configuration is present in total
        p0s = np.array([np.prod( [1-ni[i] for i in v['atoms'] ]) for v in vertices])
        p0 = p0s.mean()
        p1 = np.mean([p0s[k]*np.sum([ni[i]/(1-ni[i]) for i in v['atoms']]) for k,v in enumerate(vertices)])
        p2 = np.mean([0.5*p0s[k]* np.sum([ ni[i]/(1-ni[i]) * np.sum([ni[j]/(1-ni[j]) if j!=i else 0 for j in v['atoms']]) for i in v['atoms']]) for k,v in enumerate(vertices)])
        p3 = np.mean([ np.sum([ (1-ni[i]) * np.prod([ni[j] if j!=i else 1 for j in v['atoms']]) for i in v['atoms']]) for v in vertices])
        p4 = np.mean([np.prod([ni[i] for i in v['atoms']]) for v in vertices])

        # combine everything and return it normalized to have a probability
        p = np.array([p0, p1, p2, p3, p4])

        probs = p/p.sum()
        log_data['monomer'] = probs[0]
        log_data['dimer'] = probs[1]
        log_data['double dimer'] = probs[2]
        log_data['triple dimer'] = probs[3]
        log_data['quadruple dimer'] = probs[4]

        return True
    
class CallbackParamsJastrow:
    '''
    Stores the parameters of the Jastrow and their derivatives as histogram in tensorboard-X
    '''
    def __init__(self, name):
        '''
        name : folder in which the infos will be written
        '''
        self.name = name
        self.writer = tbx.SummaryWriter(self.name)        
        
    def __call__(self, step, log_data, driver):   

        self.writer.add_histogram('W_Re', np.array(driver.state.parameters['kernel'].real), step)
        self.writer.add_histogram('W_Im', np.array(driver.state.parameters['kernel'].imag), step)
        
        return True


def cb_derivatives(step, log_data, driver):
	'''
	Stores the derivatives of the parameters throughout the evolution
	'''
	dp = driver._dw
	#dp = driver.ode()

	log_data['dtheta'] = dp

	return True


class cb_distr:
    def __init__(self, folder=''):
        self.folder = folder
    
    def __call__(self, step, log_data, driver):
        '''
        Returns the distribution of the state
        '''
        psi = driver._variational_state.to_array(normalize=True)
        
        plt.figure()
        plt.plot( psi.conj()*psi, ls='', marker='.')
        plt.title(f'Step : {step:.2f}')
        plt.xlabel('Index')
        plt.ylabel(r'$|\psi|^2$')
        plt.savefig(self.folder+f'_step={step:.2f}.png', dpi=200, bbox_inches='tight')
        #plt.show()
        plt.close()

        return True

def cb_dt(step, log_data, driver):
    log_data['dt'] = driver.integrator.dt

    return True


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
    
