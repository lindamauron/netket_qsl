import numpy as np

from .base import Frequency

def linear_sweep(sweep_time,fMin,fMax):
    slope = (fMax-fMin)/(4*sweep_time/5)
    offset = fMin - slope*sweep_time/5
    f = 2*np.pi*np.array([slope,offset])

    return lambda t : (f[0]*t+f[1]), f



class Constant(Frequency):
    '''
    Defines the constant frequencies schedules Ω(t) and Δ(t) for a given sweep_time.
    Allows to get both frequencies at the time we want. The functions are vectorized and can be used outside the time range [0,sweep_time].
    It is also possible to integrate and derivate the frequencies over any time interval. 

    The general aspect of the schedule is as follows :
    Omega(t) = Ωf
    Delta(t) = Δf
    ''' 
    
    def __init__(self, sweep_time=19e-3,Omega=20,Delta=0):
        '''
        sweep_time : total time of the experiment (in μs)
        Omega : value of Ω
        Delta : value of Δ
        '''
        super().__init__(sweep_time,Omega)
        self.sweep, self.f = lambda t : 2*np.pi*Delta, 2*np.pi*np.array([Delta])
        self.Δf = 2*np.pi*Delta

    
    def __repr__(self):
        return f'Constant(T={self.sweep_time}, Ω={self.Ωf/2/np.pi}, Δ={self.Δf/2/np.pi})'
    
    def Ω(self, t):
        '''
        Defines the default Ω schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Ω (in Mrad/μs)
        '''
        return self.Ωf*np.ones_like(t)
    
    def Δ(self, t):
        '''
        Defines the default Δ schedule for the time evolution over a total sweep_time
        t : time of the evolution (in μs)

        return : corresponding Δ (in Mrad/μs)
        '''
        return self.Δf*np.ones_like(t)
    
    def mean_Omega(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Ω(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Ω (rad)
        '''
        return self.Ωf
    

    def mean_Delta(self, t1, t2):
        r'''
        Does the operation $\int_t1^t2 Δ(τ) dτ$
        t1 : smallest time (μs)
        t2 : highest time (μs)

        returns : mean of Δ (rad)
        '''
        
        return self.Δf
    

