import numpy as np

from .base import Frequency

def linear_sweep(sweep_time,fMin,fMax):
    slope = (fMax-fMin)/(4*sweep_time/5)
    offset = fMin - slope*sweep_time/5
    f = 2*np.pi*np.array([slope,offset])

    return lambda t : (f[0]*t+f[1]), f



class Linear(Frequency):
    '''
    Defines the linear frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectorized and can be used outside the time range [0,sweep_time].
    It is also possible to integrate and derivate the frequencies over any time interval. 

    The general aspect of the schedule is as follows :
    Omega(t) :
            - [0,sweep_time/5] : linear increase from 0 to Omega
            - [sweep_time/5,infty] : constant at Omega
    Delta(t) :
            - [0,sweep_time/5] : constant at fMin
            - [sweep_time/5,infty] : linear increase
    ''' 
    
    def __init__(self, sweep_time,Omega=1.4,fMin=-8,fMax=9.4):
        '''
        sweep_time : total time of the experiment (in μs)
        Omega : final constant value of Omega(t>sweep_time/5)
        fMin : minimal (initial) value of Delta
        fMax : maxima (final) value of Delta
        '''
        super().__init__(sweep_time,Omega)
        self.sweep, self.f = linear_sweep(sweep_time,fMin,fMax)
        self.Δi = 2*np.pi*fMin
        self.Δf = 2*np.pi*fMax

    
    def __repr__(self):
        return 'LinearSweep(' + super().__repr__()+ f'[{self.Δi/2/np.pi},{self.Δf/2/np.pi}])'

    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        flag = (t < self.sweep_time/5)
        return self.Δi*(flag) + (1 - flag)*self.sweep(t)
        

    def mean_Delta(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Δ(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Δ (rad)
        '''
        
        T = self.sweep_time/5
        
        if t2 > T:
            def F(t):
                return self.f[0]/2*t**2 + self.f[1]*t
            if t1 > T:
                return (F(t2)-F(t1) )/(t2-t1)
            else:
                return (F(t2)-F(T) + self.Δi*(T-t1))/(t2-t1)
        else:
            return self.Δi
    

