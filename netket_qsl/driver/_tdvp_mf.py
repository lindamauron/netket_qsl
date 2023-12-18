import numpy as np

from typing import Callable, Sequence

import jax.numpy as jnp
#import flax.linen as nn
from jax.config import config; config.update("jax_enable_x64", True)
from jax.tree_util import tree_map
import flax


from typing import Callable, Optional
from collections.abc import Sequence

from tqdm import tqdm

from netket import config
from netket.driver import AbstractVariationalDriver
from netket.driver.abstract_variational_driver import _to_iterable
from netket.logging.json_log import JsonLog
from netket.utils import mpi
from netket.vqs import VariationalState
from netket.stats import Stats
from netket.driver.vmc_common import info
from netket.vqs import (
    VariationalState,
)

_Nan = float("NaN")
from ._integrators import Integrator, RK4
from ..operators._hamiltonian import Rydberg_Hamiltionian as Hamiltionian


 
def z_factor(phi,t, R, freq):
    phi_plus_2 = phi[:,1].conj()*phi[:,1]
    # factor 2 disappeared bc Vij = 1/2 Omega R = Vji
    return -freq.Δ(t) + freq.Ω(t)*R@phi_plus_2

def x_factor(phi,t, R, freq):
    return -freq.Ω(t)/2

def bR_dot(phi,t, R, freq):
    a = phi[:,0]
    bR = phi[:,1].real
    bI = phi[:,1].imag

    z_term = bI
    x_term = -bI*bR/a

    return z_factor(phi,t, R, freq)*z_term + x_factor(phi,t, R, freq)*x_term

def bI_dot(phi,t, R, freq):
    a = phi[:,0]
    bR = phi[:,1].real
    bI = phi[:,1].imag

    z_term = -bR

    x_term = -a + bR**2/a

    return z_factor(phi,t, R, freq)*z_term + x_factor(phi,t, R, freq)*x_term



## Measurements
def eval_N(phi,R):
    a = phi[:,0]
    b = phi[:,1]

    return np.sum(b.conj()*b)

def eval_O(phi,R):

    a = phi[:,0]
    b = phi[:,1]

    X_term = -0.5*np.sum(2*a*b.real)
    b2 = (b.conj()*b)
    Z_term = 0.5* b2.T@R@b2
    return X_term + Z_term

def eval_N2(phi):
    N = phi.shape[0]
    a = phi[:,0]
    b = phi[:,1]

    ni = (b.conj()*b ).real

    N2 = 0
    for i in range(N):
        N2 += (ni[i] * ni[i+1:]).sum()
    
    return ni.sum()/N + 2*N2/N**2 - ni.sum()**2/N**2


class TDVP_MF(AbstractVariationalDriver):
    """
    Variational time evolution based on the time-dependent variational principle solved analytically for a mean-field ansatz.
    The solution being analytical, it can only be used for Rydberg hamiltonians. 

    .. note::
        This TDVP Driver only works for ansatzes with an external mean-field part. It will put all other parameters to zero. 
        The integrator in use is RK4. 

    """

    def __init__(
        self,
        operator: Hamiltionian,
        variational_state: VariationalState,
        integrator: Integrator=RK4(1e-3),
        *,
        t0: float = 0.0,
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (RydbergHamiltonian).
            variational_state: The variational state.
            integrator: the integrator of the ode for the parameters
            t0: Initial time at the start of the time evolution.
        """

        assert np.isin('ϕ', np.fromiter(variational_state.parameters.keys(), dtype='<U3' ) ), "The variational state must have an external mena-field with parameters defined as 'ϕ'"
        params = flax.core.unfreeze(tree_map(lambda x: 0*x, variational_state.parameters))
        N = params['ϕ'].shape[0]
        phi = jnp.zeros((N,2), dtype=variational_state.model.param_dtype)
        phi = phi.at[:,0].set(1)
        phi = phi.at[:,1].set(0)
        params['ϕ'] = phi
        variational_state.parameters = params
        variational_state.reset()
        
        self.generator = operator
        self._generator_repr = repr(operator)
        self.t = t0
        self._step_count = 0
        
        super().__init__(
            variational_state, optimizer=None, minimized_quantity_name="Generator"
        ) 
        self._loss_stats = self._forward(self.t)
        self._integrator = integrator

 
    def __repr__(self):
        return (
            "TDVP_MF("
            + f"\n  time = {self.t},"
            + f"\n  state = {self.state}"
            +"\n)"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("generator    ", self._generator_repr),
                ("integrator      ", self._integrator),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)

    def _forward(self,t):
        """
        computes the forward pass, i.e. calculates the energy of the state
        since we have a MF state, this can be exactly done
        """
        phi = self.state.parameters['ϕ']

        O_term = eval_O(phi,self.generator.R)
        N_term = eval_N(phi,self.generator.R)

        O,D = self.generator.frequencies(t)

        mean_E = O*O_term - D*N_term + 0j
        #self._loss_stats = Stats(mean=mean_E, variance=_Nan, error_of_mean=0, tau_corr=0, R_hat=0, tau_corr_max=0)

        return Stats(mean=mean_E, variance=0, error_of_mean=0, tau_corr=0, R_hat=0, tau_corr_max=0)
    
    def _backward(self,t,phi):
        #pars = self.state.parameters
        #phi = pars['ϕ']
        a = phi[:,0]
        bR = phi[:,1].real
        bI = phi[:,1].imag

        dbR = bR_dot(phi,t,self.generator.R, self.generator.frequencies)
        dbI = bI_dot(phi,t,self.generator.R, self.generator.frequencies)
        da = -(bR*dbR + bI*dbI)/a

        dphi = np.array([ da, dbR + 1j*dbI ]).T

        #dtheta = flax.core.unfreeze(tree_map(lambda x: 0*x, pars))
        #dtheta['ϕ'] = dphi


        return dphi #flax.core.FrozenDict(dtheta)

    
    def iter(self, T: float, *, tstops: Optional[Sequence[float]] = None):
        """
        Returns a generator which advances the time evolution for an interval
        of length :code:`T`, stopping at :code:`tstops`.

        Args:
            T: Length of the integration interval.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
                at which this method will stop and yield. By default, a stop is performed
                after each time step (at potentially varying step size if an adaptive
                integrator is used).
        Yields:
            The current step count.
        """
        yield from self._iter(T, tstops)
    
    def _iter(
        self,
        T: float,
        tstops: Optional[Sequence[float]] = None,
        callback: Callable = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(f"All tstops must be in range [t, t + T]=[{self.t}, {T}]")

        if tstops is not None and len(tstops) > 0:
            tstops = np.sort(tstops)
            always_stop = False
        else:
            tstops = []
            always_stop = True

        while self.t < t_end:
            if always_stop or (
                len(tstops) > 0
                and (np.isclose(self.t, tstops[0]) or self.t > tstops[0])
            ):
                yield self.t
                tstops = tstops[1:]

            step_accepted = False
            while not step_accepted:
                if not always_stop and len(tstops) > 0:
                    max_dt = tstops[0] - self.t
                else:
                    max_dt = None

                    
                step_accepted = integrate(self)
            self._loss_stats = self._forward(self.t)

            self._step_count += 1
            # optionally call callback
            if callback:
                callback()

        # Yield one last time if the remaining tstop is at t_end
        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t

    
    def run(
        self,
        T,
        out=None,
        obs=None,
        *,
        tstops=None,
        show_progress=True,
        callback=None,
    ):
        """
        Runs the time evolution.

        By default uses :class:`netket.logging.JsonLog`. To know about the output format
        check it's documentation. The logger object is also returned at the end of this function
        so that you can inspect the results without reading the json output.

        Args:
            T: The integration time period.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing the observables that should be computed.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
                at which the driver will stop and perform estimation of observables, logging, and execute
                the callback function. By default, a stop is performed after each time step (at potentially
                varying step size if an adaptive integrator is used).
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to be executed at each
                stopping time.
        """
        self.compute_n = False
        if obs is None:
            obs = {}
        elif np.isin('n', np.fromiter(obs.keys(), dtype='<U3') ):
            self.compute_n = True
            obs = obs.copy()
            obs.pop('n')

        if callback is None:
            callback = lambda *_args, **_kwargs: True

        # Log only non-root nodes
        if self._mynode == 0:
            if out is None:
                loggers = ()
            # if out is a path, create an overwriting Json Log for output
            elif isinstance(out, str):
                loggers = (JsonLog(out, "w",save_params=False),)
            else:
                loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        if isinstance(out, str):
            callbacks = callbacks + (callback_pars(out), )
        else:
            callback = callbacks + (callback_pars(''), )

        callback_stop = False

        t_end = np.asarray(self.t + T)
        with tqdm(
            total=t_end,
            disable=not show_progress,
            unit_scale=True,
            dynamic_ncols=True,
        ) as pbar:
            first_step = True

            # We need a closure to pass to self._iter in order to update the progress bar even if
            # there are no tstops
            def update_progress_bar():
                # Reset the timing of tqdm after the first step to ignore compilation time
                nonlocal first_step
                if first_step:
                    first_step = False
                    pbar.unpause()

                pbar.n = min(np.asarray(self.t), t_end)
                self._postfix["n"] = self.step_count
                self._postfix.update(
                    {
                        self._loss_name: str(self._loss_stats),
                    }
                )

                pbar.set_postfix(self._postfix)
                pbar.refresh()

            for step in self._iter(T, tstops=tstops, callback=update_progress_bar):
                log_data = self.estimate(obs)

                self._log_additional_data(log_data, self.t)

                self._postfix = {"n": self.step_count}
                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    self._postfix.update(
                        {
                            self._loss_name: str(self._loss_stats),
                        }
                    )
                    log_data[self._loss_name] = self._loss_stats
                pbar.set_postfix(self._postfix)

                # Execute callbacks before loggers because they can append to log_data
                for callback in callbacks:
                    if not callback(step, log_data, self):
                        callback_stop = True

                for logger in loggers:
                    logger(float(self.t), log_data, self.state)

                if len(callbacks) > 0:
                    if mpi.mpi_any(callback_stop):
                        break
                update_progress_bar()

            # Final update so that it shows up filled.
            update_progress_bar()

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        return loggers
    
    
    def _log_additional_data(self, log_dict, step):
        if self.compute_n:
            phi = self.state.parameters['ϕ']
            N = phi.shape[0]
            n_mean = eval_N(phi,self.generator.R)/N +0j
            # n_var = eval_N2(phi)

            log_dict['n'] = Stats(mean=n_mean, variance=0, error_of_mean=0, tau_corr=0, R_hat=0, tau_corr_max=0)

        log_dict["t"] = self.t

        return 



def integrate(driver):
    pars = driver.state.parameters
    y = pars['ϕ']

    flag, y = driver._integrator(driver,y)
    if not flag:
        return False
    
    pars = flax.core.unfreeze(tree_map(lambda x: 0*x, pars))
    pars['ϕ'] = y

    driver.state.parameters = flax.core.FrozenDict(pars)
    driver.state.reset()
    driver.t += driver._integrator.h

    return True



class callback_pars:
    def __init__(self,folder=''):
        self.folder = folder
        self.times = []
        self.pars = []

    def __call__(self,step, log_data, driver):
        self.times.append(step)
        np.save(self.folder+'time.npy', self.times)

        self.pars.append( driver.state.parameters['ϕ'] )
        np.save(self.folder+'pars.npy', self.pars)

        return True