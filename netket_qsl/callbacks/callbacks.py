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
from ..lattice import Kagome as _Lattice
from ..driver import TDVP_MF as _TDVPMF
from netket.experimental import TDVP as _TDVP

#######################################################################################################################
################################################## Callback functions #################################################
#######################################################################################################################


def cb_params(step, log_data, driver):
    pars = driver.state.parameters
    log_data['pars'] = np.fromiter(pars.values(), dtype=float)

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

