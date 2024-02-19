import numpy as np
import netket as nk

from ..operators import dimer_probs
from ..lattice import Kagome as _Lattice
from ..driver import TDVP_MF as _TDVPMF
from netket.experimental import TDVP as _TDVP
from netket.vqs import MCState as _MCState
from netket.stats import Stats

#######################################################################################################################
################################################## Callback functions #################################################
#######################################################################################################################

# def dimer_probs_sampled(step,log_data,driver,lattice):
#     probs = dimer_probs(lattice, driver.state.samples)
#     log_data['monomer'] = probs[0]
#     log_data['dimer'] = probs[1]
#     log_data['double dimer'] = probs[2]
#     log_data['triple dimer'] = probs[3]
#     log_data['quadruple dimer'] = probs[4]

#     return True

def dimer_probs_mf(step,log_data,phi,op):
    a = phi[:,0]
    b = phi[:,1]

    ni = (b.conj()*b).real

    #n = len(lattice.non_border) #this container was different for each type of lattice
    # vertices = lattice.vertices[lattice.non_border]

    # find out how many of each configuration is present in total
    p0s = np.mean([np.prod( [1-ni[i] for i in v['atoms'] ]) for v in op.v])
    p0 = p0s.mean()
    p1 = np.mean([p0s[k]*np.sum([ni[i]/(1-ni[i]) for i in v['atoms']]) for k,v in enumerate(op.v)])
    p2 = np.mean([0.5*p0s[k]* np.sum([ ni[i]/(1-ni[i]) * np.sum([ni[j]/(1-ni[j]) if j!=i else 0 for j in v['atoms']]) for i in v['atoms']]) for k,v in enumerate(op.v)])
    p3 = np.mean([ np.sum([ (1-ni[i]) * np.prod([ni[j] if j!=i else 1 for j in v['atoms']]) for i in v['atoms']]) for v in op.v])
    p4 = np.mean([np.prod([ni[i] for i in v['atoms']]) for v in op.v])

    # combine everything and return it normalized to have a probability
    p = np.array([p0, p1, p2, p3, p4])
    probs = p/p.sum()


    log_data['monomer'] = Stats(mean=probs[0], error_of_mean=0, variance=0)
    log_data['dimer'] = Stats(mean=probs[1], error_of_mean=0, variance=0)
    log_data['double dimer'] = Stats(mean=probs[2], error_of_mean=0, variance=0)
    log_data['triple dimer'] = Stats(mean=probs[3], error_of_mean=0, variance=0)
    log_data['quadruple dimer'] = Stats(mean=probs[4], error_of_mean=0, variance=0)

    return True


class callback_dimerprobs:
    r"""
    Computes the dimer (monomer,dimer,double dimer, triple dimer, quadruple dimer) probabilities of a variational state

    If the driver is mean-field, this means that the probabilities can be computed exactly. 
    Otherwise, we do it by sampling the state. 
    """
    def __init__(self,operator=None):
        self._operator = operator

    @property
    def operator(self):
        return self._operator

    def __call__(self, step, log_data, driver):
        if isinstance(driver, _TDVP):
            probs = driver.state.expect( self._operator )

            log_data['monomer'] = probs['monomer']
            log_data['dimer'] = probs['dimer']
            log_data['double dimer'] = probs['double dimer']
            log_data['triple dimer'] = probs['triple dimer']
            log_data['quadruple dimer'] = probs['quadruple dimer']

            return True
            # return dimer_probs_sampled(step,log_data,driver,self.lattice)
        elif isinstance(driver,_TDVPMF):
            phi = driver.state.parameters['ϕ']
            return dimer_probs_mf(step,log_data,phi,self._operator)
        
        elif isinstance(driver,_MCState):
            probs = driver.expect( self._operator )

            log_data['monomer'] = probs['monomer']
            log_data['dimer'] = probs['dimer']
            log_data['double dimer'] = probs['double dimer']
            log_data['triple dimer'] = probs['triple dimer']
            log_data['quadruple dimer'] = probs['quadruple dimer']

            return True
        

        return False
