import numpy as np

from .base import Frequency

def negative_quadratic_sweep(sweep_time,fMin,fMax):
    c = (25*fMin - 9*fMax)/16
    a = (c-fMax)/sweep_time**2
    b = -2*a*sweep_time

    f = 2*np.pi*np.array([a,b,c])

    return lambda t : (f[0]*t**2+f[1]*t+f[2]), f

def positive_quadratic_sweep(sweep_time,fMin,fMax):
    a = (fMax-fMin)/sweep_time**2 * 25/16
    b = -2*a*sweep_time/5
    c = (fMax + 15*fMin)/16

    f = 2*np.pi*np.array([a,b,c])

    return lambda t : f[0]*t**2+f[1]*t+f[2], f


class Quadratic(Frequency):
    '''
    Defines the linear frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectorized and can be used outside the time range [0,sweep_time] But it is constant.
    It is also possible to integrate and derivate the frequencies over any time interval. 

    The general aspect of the schedule is as follows :
    Omega(t) :
            - [0,sweep_time/5] : linear increase from 0 to Omega
            - [sweep_time/5,infty] : constant at Omega
    Delta(t) :
            - [0,sweep_time/5] : constant at fMin
            - [sweep_time/5,sweep_time] : quadratic increase
            - [sweep_time,infty] : constant at fMax
    ''' 
    
    def __init__(self, sweep_time,Omega=1.4,fMin=-8,fMax=9.4, sign="negative"):
        '''
        sweep_time : total time of the experiment (in μs)
        Omega : final constant value of Omega(t>sweep_time/5)
        fMin : minimal (initial) value of Delta
        fMax : maxima (final) value of Delta
        sign : gives the sign of the parabola sign(a), which decides on what point the extrema of Δ is located. 
        '''
        super().__init__(sweep_time,Omega)
        if sign=="negative":
            self.sweep, self.f = negative_quadratic_sweep(sweep_time,fMin,fMax)
        elif sign=="positive":
            self.sweep, self.f = positive_quadratic_sweep(sweep_time,fMin,fMax)
        else:
            raise AttributeError(f"The sign of the parabola must be given as a string, instead got {sign}. ")
        self.Δi = 2*np.pi*fMin
        self.Δf = 2*np.pi*fMax

    
    def __repr__(self):
        return 'QuadraticSweep(' + super().__repr__()+ f'[{self.Δi/2/np.pi},{self.Δf/2/np.pi}])'

    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return self.Δi*(flag) + (1 - flag)*self.sweep(np.clip(t,a_min=0,a_max=self.sweep_time))
        

    def mean_Delta(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Δ(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Δ (rad)
        '''
        if t1>self.sweep_time or t2>self.sweep_time:
            raise ValueError(f"The integration of Delta is not implemented for times greater than sweep-time, got T={self.sweep_time}, t1={t1}, t2={t2}")

        T = self.sweep_time/5

        if t2 > T:
            def F(t):
                return self.f[0]/3*t**3 + self.f[1]/2*t**2 + self.f[0]*t
            if t1 > T:
                return (F(t2)-F(t1) )/(t2-t1)
            else:
                return (F(t2)-F(T) + self.Δi*(T-t1))/(t2-t1)
        else:
            return self.Δi
    